#pragma once

#include <cell_complex/Cell.h>
#include <cell_complex/CellComplex.h>
#include <cell_complex/logging.h>
#include <cell_complex/simplicial_columns.h>
#include <cell_complex/simplicial_complex.h>
#include <cell_complex/triangulation.h>
#include <cell_complex/dominance.h>

#include <ankerl/unordered_dense.h>
#include <mshio/mshio.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cell_complex {

/// @brief Options for saving MSH files
struct SaveMshOptions
{
    bool binary = true; ///< Use binary format (true) or ASCII (false)
    bool save_element_data = true; ///< Save 3-cell membership as element data
    std::string data_field_name = "volume_id"; ///< Name for the element data field
    bool save_sign_data = true; ///< Save sign history as element data
    bool save_vertex_values = true; ///< Save Cell<0>::value as $NodeData (one scalar per node)
    std::string vertex_value_field_name = "vertex_value"; ///< Name for the node data field
    bool save_label_data = true; ///< Save Cell<D>::label as element data
    std::string label_field_name = "label"; ///< Name for the label field
    std::string dominant_leaf_field_name = "dominant_leaf"; ///< Name for the dominant leaf field
    size_t dominant_leaf_root_bit = 0; ///< Start bit of the root chunk in the dominance mask
    size_t dominant_leaf_num_leafs = 0; ///< Number of leaves (chunk size); 0 = skip dominant leaf field
};

/// @brief Options for loading MSH files
struct LoadMshOptions
{
    bool verbose = false; ///< Print information about loaded mesh
};

namespace detail {

/// @brief Build a mapping from vertex keys to sequential 1-based node tags
template <int DIM>
ankerl::unordered_dense::map<size_t, size_t> build_vertex_tag_map(CellComplex<DIM>& cc)
{
    ankerl::unordered_dense::map<size_t, size_t> vertex_key_to_tag;
    auto& vertices = cc.template get_cells<0>();

    size_t tag = 1; // MSH node tags are 1-based
    for (const auto& [key, vertex_ref] : vertices.items()) {
        auto vertex_index = get_index<0, DIM>(key);
        vertex_key_to_tag[vertex_index] = tag++;
    }

    return vertex_key_to_tag;
}

/// @brief Extract node coordinates from cell complex.
/// MSH supports only 3 spatial dimensions per vertex. For DIM > 3, only the
/// first 3 coordinates are exported; higher-dimensional coordinates (e.g. time
/// for DIM == 4) are exported as a separate $NodeData field.
template <int DIM>
std::vector<double> extract_node_coordinates(CellComplex<DIM>& cc)
{
    auto& vertices = cc.template get_cells<0>();
    std::vector<double> coords;
    coords.reserve(vertices.size() * 3);

    constexpr int spatial_dim = (DIM < 3) ? DIM : 3;
    for (const auto& [key, vertex_ref] : vertices.items()) {
        const auto& vertex = vertex_ref.get();
        for (int d = 0; d < spatial_dim; ++d) {
            coords.push_back(vertex.coordinates[d]);
        }
        for (int d = spatial_dim; d < 3; ++d) {
            coords.push_back(0.0);
        }
    }

    return coords;
}

/// @brief Extract the time coordinate (last dimension) per vertex for DIM == 4.
template <int DIM>
std::vector<double> extract_vertex_time_values(CellComplex<DIM>& cc)
{
    static_assert(DIM == 4, "extract_vertex_time_values is only defined for DIM == 4");
    auto& vertices = cc.template get_cells<0>();
    std::vector<double> times;
    times.reserve(vertices.size());
    for (const auto& [key, vertex_ref] : vertices.items()) {
        times.push_back(static_cast<double>(vertex_ref.get().coordinates[3]));
    }
    return times;
}

/// @brief Extract optional 0-cell scalar values in the same order as nodes (and extract_node_coordinates)
template <int DIM>
std::vector<double> extract_vertex_values(CellComplex<DIM>& cc)
{
    auto& vertices = cc.template get_cells<0>();
    std::vector<double> values;
    values.reserve(vertices.size());

    for (const auto& [key, vertex_ref] : vertices.items()) {
        values.push_back(static_cast<double>(vertex_ref.get().value));
    }

    return values;
}

/// @brief Get the two vertex tags for an edge (1-cell)
template <int DIM>
std::array<size_t, 2> get_edge_vertex_tags(
    CellComplex<DIM>& cc,
    const typename Cell<1, DIM>::KeyType& edge_key,
    const ankerl::unordered_dense::map<size_t, size_t>& vertex_key_to_tag)
{
    auto& edge = cc.template get_cell<1>(edge_key);

    if (edge.boundary.size() != 2) {
        throw std::runtime_error("Edge must have exactly 2 vertices");
    }

    // Get the directed endpoints (tail, head) of this oriented edge
    auto [tail, head] = cell_complex::detail::get_edge_directed_endpoints<DIM>(cc, edge_key);
    auto tail_index = get_index<0, DIM>(tail);
    auto head_index = get_index<0, DIM>(head);

    auto it_tail = vertex_key_to_tag.find(tail_index);
    if (it_tail == vertex_key_to_tag.end()) {
        throw std::runtime_error("Tail vertex not found in tag map");
    }

    auto it_head = vertex_key_to_tag.find(head_index);
    if (it_head == vertex_key_to_tag.end()) {
        throw std::runtime_error("Head vertex not found in tag map");
    }

    return {it_tail->second, it_head->second};
}

/// @brief Get the three vertex tags for a triangular 2-cell in boundary-cycle order
///
/// The returned node order follows the oriented boundary cycle of the 2-cell,
/// with the orientation tag on `face_key` applied: a `Positive` tag preserves
/// the 2-cell's intrinsic winding; a `Negative` tag reverses it (swap node[1]
/// and node[2]). This ensures that when used as a boundary face of a 3-cell,
/// the emitted MSH triangle's normal matches the 3-cell's outward convention.
///
/// `face.boundary` is not required to be stored in cycle order — the cycle is
/// reconstructed via `build_ordered_boundary_cycle` before extracting tails.
template <int DIM>
std::array<size_t, 3> get_triangle_vertex_tags(
    CellComplex<DIM>& cc,
    const typename Cell<2, DIM>::KeyType& face_key,
    const ankerl::unordered_dense::map<size_t, size_t>& vertex_key_to_tag)
{
    auto& face = cc.template get_cell<2>(face_key);

    if (face.boundary.size() != 3) {
        throw std::runtime_error("Face must be triangular (have exactly 3 boundary edges)");
    }

    auto ordered = cell_complex::detail::build_ordered_boundary_cycle<DIM>(cc, face.boundary);
    assert(ordered.size() == 3);

    std::array<size_t, 3> vertex_tags;
    for (size_t i = 0; i < 3; ++i) {
        auto [tail, head] = cell_complex::detail::get_edge_directed_endpoints<DIM>(cc, ordered[i]);
        auto it = vertex_key_to_tag.find(get_index<0, DIM>(tail));
        if (it == vertex_key_to_tag.end()) {
            throw std::runtime_error("Vertex not found in tag map");
        }
        vertex_tags[i] = it->second;
    }

    if (get_orientation<2, DIM>(face_key) == Orientation::Negative) {
        std::swap(vertex_tags[1], vertex_tags[2]);
    }

    return vertex_tags;
}

} // namespace detail

/// @brief Save a cell complex to MSH format
/// @param filename Path to the output MSH file
/// @param cc The cell complex to save
/// @param options Options controlling the output format
///
/// The function detects the highest-dimensional cells present and saves them:
/// - If highest dimension is 1: saves edges (1-cells) as line elements
/// - If highest dimension is 2: saves triangles (2-cells) as triangle elements
/// - If highest dimension is 3+: saves boundary triangles of 3-cells
///
/// Polygonal 2-cells are triangulated on the local copy via triangulate_all_2cells before export.
/// An element data field indicates which higher-dimensional cell each element belongs to.
/// Optional $NodeData section stores each vertex's scalar `Cell<0,DIM>::value` (see Cell.h).
///
/// @throws std::runtime_error if the cell complex is invalid or writing fails
template <int DIM>
void save_msh(
    const std::string& filename,
    CellComplex<DIM> cc,
    const SaveMshOptions& options = SaveMshOptions{})
{
    static_assert(DIM >= 1, "save_msh requires at least 1D cell complexes");

    auto logger = get_logger();
    logger->debug("Saving {}D cell complex to MSH file: {}", DIM, filename);
    logger->debug(
        "  Format: {}, Element data: {}, Sign data: {}, Vertex values: {}",
        options.binary ? "binary" : "ASCII",
        options.save_element_data,
        options.save_sign_data,
        options.save_vertex_values);

    // Determine the highest dimension cell present
    const bool has_2_cells = [&]() {
        if constexpr (DIM >= 2) {
            return cc.template num_cells<2>() > 0;
        }
        return false;
    }();

    const bool has_3_cells = [&]() {
        if constexpr (DIM >= 3) {
            return cc.template num_cells<3>() > 0;
        }
        return false;
    }();

    const bool has_higher_cells = [&]() {
        if constexpr (DIM >= 4) {
            // For now, assume DIM <= 4 in practice
            // Could be extended with template recursion if needed
        }
        return false;
    }();

    // Determine if we should save edges (1-cells) or triangles (2-cells)
    const bool save_edges = !has_2_cells && !has_3_cells && !has_higher_cells;

    // Triangulate all polygonal 2-cells upfront (only if we're saving 2-cells).
    // Centroid-fan triangulation adds new 0-cells (centroids); the vertex tag map
    // must be built afterward so every node — including centroids — is assigned an MSH tag.
    if (!save_edges) {
        if constexpr (DIM >= 2) {
            triangulate_all_2cells(cc);
        }
    }

    // Step 2: Build vertex mapping
    auto vertex_key_to_tag = detail::build_vertex_tag_map<DIM>(cc);

    // Step 3: Extract node coordinates
    auto node_coords = detail::extract_node_coordinates<DIM>(cc);

    // Step 4: Prepare MshSpec
    mshio::MshSpec spec;

    // Mesh format section
    spec.mesh_format.version = "4.1";
    spec.mesh_format.file_type = options.binary ? 1 : 0;
    spec.mesh_format.data_size = sizeof(size_t);

    // Entities section (required for MSH 4.1)
    // For edge meshes: create a curve entity; otherwise create a surface entity
    if (save_edges) {
        // Create a single curve entity for all edges
        mshio::CurveEntity curve_entity;
        curve_entity.tag = 1;
        curve_entity.min_x = -1e10;
        curve_entity.min_y = -1e10;
        curve_entity.min_z = -1e10;
        curve_entity.max_x = 1e10;
        curve_entity.max_y = 1e10;
        curve_entity.max_z = 1e10;
        // boundary_point_tags is empty (vector)
        // physical_group_tags is empty (vector)
        spec.entities.curves.push_back(curve_entity);
    } else {
        // Create a single surface entity for all triangles
        mshio::SurfaceEntity surface_entity;
        surface_entity.tag = 1;
        surface_entity.min_x = -1e10;
        surface_entity.min_y = -1e10;
        surface_entity.min_z = -1e10;
        surface_entity.max_x = 1e10;
        surface_entity.max_y = 1e10;
        surface_entity.max_z = 1e10;
        // boundary_curve_tags is empty (vector)
        // physical_group_tags is empty (vector)
        spec.entities.surfaces.push_back(surface_entity);
    }

    // Nodes section
    const size_t num_vertices = cc.template num_cells<0>();
    spec.nodes.num_entity_blocks = 1;
    spec.nodes.num_nodes = num_vertices;
    spec.nodes.min_node_tag = (num_vertices > 0) ? 1 : 0;
    spec.nodes.max_node_tag = num_vertices;

    mshio::NodeBlock node_block;
    node_block.entity_dim =
        save_edges ? 1 : 2; // Nodes belong to curve (edges) or surface (triangles) entity
    node_block.entity_tag = 1;
    node_block.parametric = 0;
    node_block.num_nodes_in_block = num_vertices;

    // Fill node tags (1, 2, 3, ...)
    node_block.tags.resize(num_vertices);
    for (size_t i = 0; i < num_vertices; ++i) {
        node_block.tags[i] = i + 1;
    }

    node_block.data = std::move(node_coords);
    spec.nodes.entity_blocks.push_back(std::move(node_block));

    // Step 5: Extract elements (edges or triangles) and build metadata
    std::vector<size_t> element_data_flat; // Element tags and node tags
    std::vector<int>
        element_to_parent_id; // For element data (volume_id for triangles, -1 for edges/faces)
    std::vector<std::vector<Sign>> element_sign_history; // Sign history for each element
    std::vector<uint8_t> element_label;
    std::vector<int> element_dominant_leaf; // Per-element dominant leaf index (-1 if none)

    size_t element_tag = 1; // MSH element tags are 1-based

    // Extract edges (1-cells) if that's the highest dimension
    if (save_edges) {
        auto& edges = cc.template get_cells<1>();

        for (auto& [edge_key, edge_ref] : edges.items()) {
            auto& edge = edge_ref.get();
            // Set orientation to Positive for iteration (iterator keys have Unknown orientation)
            auto oriented_edge_key = set_orientation<1, DIM>(edge_key, Orientation::Positive);
            auto vertex_tags =
                detail::get_edge_vertex_tags<DIM>(cc, oriented_edge_key, vertex_key_to_tag);

            element_data_flat.push_back(element_tag);
            element_data_flat.push_back(vertex_tags[0]);
            element_data_flat.push_back(vertex_tags[1]);

            element_to_parent_id.push_back(-1);
            element_sign_history.push_back(
                std::vector<Sign>(edge.cut_signs.begin(), edge.cut_signs.end()));
            element_label.push_back(edge.label);
            element_dominant_leaf.push_back(find_dominant_leaf(
                get_cell_dominance<1>(cc, edge),
                options.dominant_leaf_root_bit,
                options.dominant_leaf_num_leafs));
            ++element_tag;
        }
    }
    // Extract triangles (2-cells) otherwise
    else {
        if constexpr (DIM >= 3) {
            if (has_3_cells) {
                // 3D volume mesh: boundaries were updated by triangulate_all_2cells
                auto& volumes = cc.template get_cells<3>();
                size_t volume_index = 0;

                for (auto& [vol_key, vol_ref] : volumes.items()) {
                    auto& volume = vol_ref.get();
                    int vol_leaf = find_dominant_leaf(
                        get_cell_dominance<3>(cc, volume),
                        options.dominant_leaf_root_bit,
                        options.dominant_leaf_num_leafs);

                    for (const auto& face_key : volume.boundary) {
                        auto vertex_tags =
                            detail::get_triangle_vertex_tags<DIM>(cc, face_key, vertex_key_to_tag);

                        element_data_flat.push_back(element_tag);
                        element_data_flat.push_back(vertex_tags[0]);
                        element_data_flat.push_back(vertex_tags[1]);
                        element_data_flat.push_back(vertex_tags[2]);

                        element_to_parent_id.push_back(static_cast<int>(volume_index));

                        const auto& signs = volume.cut_signs;
                        element_sign_history.push_back(
                            std::vector<Sign>(signs.begin(), signs.end()));

                        const auto& bd_face = cc.template get_cell<2>(face_key);
                        element_label.push_back(bd_face.label);
                        element_dominant_leaf.push_back(vol_leaf);

                        ++element_tag;
                    }

                    ++volume_index;
                }
            } else {
                // Surface mesh in 3D (or 2D graph with no 2-cells - but that's handled by
                // save_edges)
                auto& faces = cc.template get_cells<2>();

                for (auto& [face_key, face_ref] : faces.items()) {
                    auto& face = face_ref.get();
                    auto vertex_tags =
                        detail::get_triangle_vertex_tags<DIM>(cc, face_key, vertex_key_to_tag);

                    element_data_flat.push_back(element_tag);
                    element_data_flat.push_back(vertex_tags[0]);
                    element_data_flat.push_back(vertex_tags[1]);
                    element_data_flat.push_back(vertex_tags[2]);

                    element_to_parent_id.push_back(-1);
                    element_sign_history.push_back(
                        std::vector<Sign>(face.cut_signs.begin(), face.cut_signs.end()));
                    element_label.push_back(face.label);
                    element_dominant_leaf.push_back(find_dominant_leaf(
                        get_cell_dominance<2>(cc, face),
                        options.dominant_leaf_root_bit,
                        options.dominant_leaf_num_leafs));
                    ++element_tag;
                }
            }
        } else if constexpr (DIM >= 2) {
            // 2D surface mesh
            auto& faces = cc.template get_cells<2>();

            for (auto& [face_key, face_ref] : faces.items()) {
                auto& face = face_ref.get();
                auto vertex_tags =
                    detail::get_triangle_vertex_tags<DIM>(cc, face_key, vertex_key_to_tag);

                element_data_flat.push_back(element_tag);
                element_data_flat.push_back(vertex_tags[0]);
                element_data_flat.push_back(vertex_tags[1]);
                element_data_flat.push_back(vertex_tags[2]);

                element_to_parent_id.push_back(-1);
                element_sign_history.push_back(
                    std::vector<Sign>(face.cut_signs.begin(), face.cut_signs.end()));
                element_label.push_back(face.label);
                element_dominant_leaf.push_back(find_dominant_leaf(
                    get_cell_dominance<2>(cc, face),
                    options.dominant_leaf_root_bit,
                    options.dominant_leaf_num_leafs));
                ++element_tag;
            }
        }
    }

    // Step 6: Create Elements section
    size_t num_elements = element_tag - 1;

    spec.elements.num_entity_blocks = 1;
    spec.elements.num_elements = num_elements;
    spec.elements.min_element_tag = 1;
    spec.elements.max_element_tag = num_elements;

    mshio::ElementBlock element_block;
    if (save_edges) {
        element_block.entity_dim = 1; // Curve entity
        element_block.entity_tag = 1;
        element_block.element_type = 1; // Line (2-node)
    } else {
        element_block.entity_dim = 2; // Surface entity
        element_block.entity_tag = 1;
        element_block.element_type = 2; // Triangle
    }
    element_block.num_elements_in_block = num_elements;
    element_block.data = std::move(element_data_flat);

    spec.elements.entity_blocks.push_back(std::move(element_block));

    // Step 7: Add node data (0-cell value field) if requested
    if (options.save_vertex_values && num_vertices > 0) {
        mshio::Data node_value_data;

        node_value_data.header.string_tags = {options.vertex_value_field_name};
        node_value_data.header.real_tags = {};
        node_value_data.header.int_tags = {
            0,
            1,
            static_cast<int>(num_vertices)}; // time_step, num_components, num entities (nodes)

        auto vertex_values = detail::extract_vertex_values<DIM>(cc);

        node_value_data.entries.resize(num_vertices);
        for (size_t i = 0; i < num_vertices; ++i) {
            node_value_data.entries[i].tag = i + 1; // node tags are 1-based
            node_value_data.entries[i].data = {vertex_values[i]};
        }

        spec.node_data.push_back(std::move(node_value_data));
    }

    // Step 7b: For DIM == 4, export the 4th coordinate as a per-vertex "time" field
    // (MSH vertices only carry 3 spatial coordinates).
    if constexpr (DIM == 4) {
        if (num_vertices > 0) {
            mshio::Data time_data;
            time_data.header.string_tags = {"time"};
            time_data.header.real_tags = {};
            time_data.header.int_tags = {0, 1, static_cast<int>(num_vertices)};

            auto times = detail::extract_vertex_time_values<DIM>(cc);

            time_data.entries.resize(num_vertices);
            for (size_t i = 0; i < num_vertices; ++i) {
                time_data.entries[i].tag = i + 1;
                time_data.entries[i].data = {times[i]};
            }
            spec.node_data.push_back(std::move(time_data));
        }
    }

    // Step 8: Add element data (volume_id field) if requested
    if (options.save_element_data && !element_to_parent_id.empty() && num_elements > 0) {
        mshio::Data elem_data;

        // Header
        elem_data.header.string_tags = {options.data_field_name};
        elem_data.header.real_tags = {}; // No time value
        elem_data.header.int_tags = {
            0,
            1,
            static_cast<int>(num_elements)}; // time_step=0, num_fields=1, num_entities

        // Data entries
        elem_data.entries.resize(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
            elem_data.entries[i].tag = i + 1; // Element tag (1-based)
            elem_data.entries[i].data = {static_cast<double>(element_to_parent_id[i])};
        }

        spec.element_data.push_back(std::move(elem_data));
    }

    // Step 9: Add sign history element data if requested
    if (options.save_sign_data && !element_sign_history.empty()) {
        // Determine the maximum number of cuts across all elements
        size_t max_cuts = 0;
        for (const auto& signs : element_sign_history) {
            max_cuts = std::max(max_cuts, signs.size());
        }

        // Create one element data field for each cut index
        for (size_t cut_idx = 0; cut_idx < max_cuts; ++cut_idx) {
            mshio::Data sign_data;

            // Header - field name is "sign_0", "sign_1", etc.
            std::string field_name = "sign_" + std::to_string(cut_idx);
            sign_data.header.string_tags = {field_name};
            sign_data.header.real_tags = {}; // No time value
            sign_data.header.int_tags = {
                0,
                1,
                static_cast<int>(num_elements)}; // time_step=0, num_fields=1, num_entities

            // Data entries
            sign_data.entries.resize(num_elements);
            for (size_t i = 0; i < num_elements; ++i) {
                sign_data.entries[i].tag = i + 1; // Element tag (1-based)

                // If this element has a sign at this cut index, use it; otherwise use 0
                if (cut_idx < element_sign_history[i].size()) {
                    sign_data.entries[i].data = {
                        static_cast<double>(static_cast<int>(element_sign_history[i][cut_idx]))};
                } else {
                    // Element doesn't have this many cuts - use 0
                    sign_data.entries[i].data = {0.0};
                }
            }

            spec.element_data.push_back(std::move(sign_data));
        }
    }

    // Step 9b: Add label element data if requested
    if (options.save_label_data && !element_label.empty() && num_elements > 0) {
        mshio::Data label_data;
        label_data.header.string_tags = {options.label_field_name};
        label_data.header.real_tags = {};
        label_data.header.int_tags = {0, 1, static_cast<int>(num_elements)};

        label_data.entries.resize(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
            label_data.entries[i].tag = i + 1;
            label_data.entries[i].data = {static_cast<double>(static_cast<int>(element_label[i]))};
        }

        spec.element_data.push_back(std::move(label_data));
    }

    // Step 9c: Add dominant leaf element data if requested
    if (options.dominant_leaf_num_leafs > 0 && !element_dominant_leaf.empty() && num_elements > 0) {
        mshio::Data leaf_data;
        leaf_data.header.string_tags = {options.dominant_leaf_field_name};
        leaf_data.header.real_tags = {};
        leaf_data.header.int_tags = {0, 1, static_cast<int>(num_elements)};

        leaf_data.entries.resize(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
            leaf_data.entries[i].tag = i + 1;
            leaf_data.entries[i].data = {static_cast<double>(element_dominant_leaf[i])};
        }

        spec.element_data.push_back(std::move(leaf_data));
    }

    // Step 10: Write to file
    logger->debug(
        "Writing MSH file with {} nodes and {} elements",
        spec.nodes.num_nodes,
        spec.elements.num_elements);
    mshio::save_msh(filename, spec);
    logger->info("Successfully saved {}D cell complex to {}", DIM, filename);
}

/// @brief Load a 3D cell complex from MSH format
/// @tparam DIM The dimension of the cell complex (2 or 3)
/// @param filename Path to the input MSH file
/// @param options Options controlling the loading behavior
/// @return CellComplex containing the loaded mesh
///
/// @throws std::runtime_error if the file cannot be read or contains unsupported element types
///
/// For DIM=2: Loads triangle elements (type 2) as 2-cells
/// For DIM=3: Loads tetrahedron elements (type 4) as 3-cells OR triangle elements (type 2) as 2-cells
///
/// The function uses from_simplicial_complex to convert the mesh data to a cell complex.
/// Only triangle and tetrahedron elements are supported for now.
template <int DIM>
CellComplex<DIM> load_msh(
    const std::string& filename,
    const LoadMshOptions& options = LoadMshOptions{})
{
    static_assert(DIM == 2 || DIM == 3, "load_msh only supports DIM=2 or DIM=3");

    // Load MSH file using mshio
    auto logger = get_logger();
    logger->debug("Loading MSH file: {}", filename);
    auto spec = mshio::load_msh(filename);

    if (options.verbose) {
        logger->info("Loading MSH file: {}", filename);
        logger->info("  Nodes: {}", spec.nodes.num_nodes);
        logger->info("  Elements: {}", spec.elements.num_elements);
    }

    // Extract vertices
    // In MSH 4.1, node tags are explicit and may be arbitrary or out of order,
    // so build a nodeTag -> contiguous vertex index map while assembling the
    // vertex buffer.
    std::vector<Scalar> vertices(spec.nodes.num_nodes * DIM);
    ankerl::unordered_dense::map<size_t, size_t> node_tag_to_index;
    node_tag_to_index.reserve(spec.nodes.num_nodes);

    size_t next_vertex_index = 0;
    for (const auto& node_block : spec.nodes.entity_blocks) {
        size_t num_nodes = node_block.num_nodes_in_block;
        const auto& tags = node_block.tags;
        const auto& data = node_block.data;

        if (tags.size() != num_nodes) {
            throw std::runtime_error("Invalid MSH node block: tag count does not match node count");
        }
        if (data.size() != num_nodes * 3) {
            throw std::runtime_error(
                "Invalid MSH node block: coordinate count does not match node count");
        }

        for (size_t i = 0; i < num_nodes; ++i) {
            const size_t node_tag = static_cast<size_t>(tags[i]);
            const size_t vertex_index = next_vertex_index++;

            // Detect duplicate node tags
            const auto [it, inserted] = node_tag_to_index.emplace(node_tag, vertex_index);
            if (!inserted) {
                throw std::runtime_error(
                    "Invalid MSH file: duplicate node tag " + std::to_string(node_tag));
            }

            // MSH stores coordinates as (x, y, z)
            for (int d = 0; d < DIM; ++d) {
                vertices[vertex_index * DIM + static_cast<size_t>(d)] =
                    static_cast<Scalar>(data[i * 3 + static_cast<size_t>(d)]);
            }
        }
    }

    // Extract elements based on dimension
    // Connectivity must be converted from MSH node tags to contiguous vertex indices
    // using node_tag_to_index.
    std::vector<size_t> tetrahedra;
    std::vector<size_t> triangles;

    for (const auto& element_block : spec.elements.entity_blocks) {
        int element_type = element_block.element_type;

        if constexpr (DIM == 2) {
            // For 2D, we expect triangles (type 2)
            if (element_type == 2) { // Triangle
                size_t num_elements = element_block.num_elements_in_block;
                const auto& data = element_block.data;

                // Each element is: [tag, v0, v1, v2]
                for (size_t i = 0; i < num_elements; ++i) {
                    size_t offset = i * 4; // tag + 3 vertices
                    // Convert node tags to vertex indices
                    triangles.push_back(node_tag_to_index.at(data[offset + 1]));
                    triangles.push_back(node_tag_to_index.at(data[offset + 2]));
                    triangles.push_back(node_tag_to_index.at(data[offset + 3]));
                }

                if (options.verbose) {
                    logger->info("  Found {} triangles", num_elements);
                }
            } else {
                if (options.verbose) {
                    logger->warn("  Skipping element type {} (not a triangle)", element_type);
                }
            }
        } else if constexpr (DIM == 3) {
            // For 3D, we support both tetrahedra (type 4) and triangles (type 2)
            if (element_type == 4) { // Tetrahedron
                size_t num_elements = element_block.num_elements_in_block;
                const auto& data = element_block.data;

                // Each element is: [tag, v0, v1, v2, v3]
                for (size_t i = 0; i < num_elements; ++i) {
                    size_t offset = i * 5; // tag + 4 vertices
                    // Convert node tags to vertex indices
                    tetrahedra.push_back(node_tag_to_index.at(data[offset + 1]));
                    tetrahedra.push_back(node_tag_to_index.at(data[offset + 2]));
                    tetrahedra.push_back(node_tag_to_index.at(data[offset + 3]));
                    tetrahedra.push_back(node_tag_to_index.at(data[offset + 4]));
                }

                if (options.verbose) {
                    logger->info("  Found {} tetrahedra", num_elements);
                }
            } else if (element_type == 2) { // Triangle (surface mesh in 3D)
                size_t num_elements = element_block.num_elements_in_block;
                const auto& data = element_block.data;

                // Each element is: [tag, v0, v1, v2]
                for (size_t i = 0; i < num_elements; ++i) {
                    size_t offset = i * 4; // tag + 3 vertices
                    // Convert node tags to vertex indices
                    triangles.push_back(node_tag_to_index.at(data[offset + 1]));
                    triangles.push_back(node_tag_to_index.at(data[offset + 2]));
                    triangles.push_back(node_tag_to_index.at(data[offset + 3]));
                }

                if (options.verbose) {
                    logger->info("  Found {} triangles (surface mesh)", num_elements);
                }
            } else {
                if (options.verbose) {
                    logger->warn(
                        "  Skipping element type {} (not a triangle or tetrahedron)",
                        element_type);
                }
            }
        }
    }

    if constexpr (DIM == 2) {
        if (triangles.empty()) {
            logger->error("No triangles found in MSH file: {}", filename);
            throw std::runtime_error("No triangles found in MSH file.");
        }
        logger->debug("Converting {} triangles to 2D cell complex", triangles.size() / 3);
        return from_simplicial_complex<2>(vertices, triangles);
    } else if constexpr (DIM == 3) {
        // For 3D, prioritize tetrahedra if available
        if (!tetrahedra.empty()) {
            if (!triangles.empty() && options.verbose) {
                logger->warn(
                    "  File contains both tetrahedra and triangles. Loading only tetrahedra.");
            }
            logger->debug("Converting {} tetrahedra to 3D cell complex", tetrahedra.size() / 4);
            return from_simplicial_complex<3>(vertices, tetrahedra);
        } else if (!triangles.empty()) {
            // Load triangles as a surface mesh (2-cells in 3D space)
            // We need to create a 3D cell complex manually since from_simplicial_complex<2>
            // would create a 2D complex
            CellComplex<3> cc;

            // Use from_simplicial_complex<2> to build topology, then embed in 3D
            // First, create 2D vertices (just x,y)
            std::vector<Scalar> vertices_2d;
            vertices_2d.reserve(vertices.size() / 3 * 2);
            for (size_t i = 0; i < vertices.size(); i += 3) {
                vertices_2d.push_back(vertices[i]);
                vertices_2d.push_back(vertices[i + 1]);
            }

            // Build 2D complex
            auto cc_2d = from_simplicial_complex<2>(vertices_2d, triangles);

            // Now manually build a 3D complex from the 2D complex
            // Map 2D vertex keys to 3D vertex keys
            ankerl::unordered_dense::map<size_t, typename Cell<0, 3>::KeyType> vertex_key_map;

            auto& vertices_2d_map = cc_2d.template get_cells<0>();
            for (const auto& [key_2d, vertex_ref_2d] : vertices_2d_map.items()) {
                Cell<0, 3> vertex_3d;
                const auto& v2d = vertex_ref_2d.get();
                auto v_idx_2d = get_index<0, 2>(key_2d);
                vertex_3d.coordinates[0] = v2d.coordinates[0];
                vertex_3d.coordinates[1] = v2d.coordinates[1];
                vertex_3d.coordinates[2] =
                    vertices[v_idx_2d * 3 + 2]; // Get z from the original vertex index
                auto key_3d = cc.add_cell<0>(vertex_3d);
                vertex_key_map[v_idx_2d] = key_3d;
            }

            // Map edges
            ankerl::unordered_dense::map<size_t, typename Cell<1, 3>::KeyType> edge_key_map;
            auto& edges_2d = cc_2d.template get_cells<1>();
            for (const auto& [key_2d, edge_ref_2d] : edges_2d.items()) {
                const auto& e2d = edge_ref_2d.get();
                Cell<1, 3> edge_3d;

                for (const auto& bd_key_2d : e2d.boundary) {
                    auto ori = get_orientation<0, 2>(bd_key_2d);
                    auto v_idx_2d = get_index<0, 2>(bd_key_2d);
                    auto v_key_3d = vertex_key_map.at(v_idx_2d);
                    edge_3d.add_boundary_cell(v_key_3d, ori);
                }

                auto key_3d = cc.add_cell<1>(edge_3d);
                edge_key_map[get_index<1, 2>(key_2d)] = key_3d;
            }

            // Map faces
            auto& faces_2d = cc_2d.template get_cells<2>();
            for (const auto& [key_2d, face_ref_2d] : faces_2d.items()) {
                const auto& f2d = face_ref_2d.get();
                Cell<2, 3> face_3d;

                for (const auto& bd_key_2d : f2d.boundary) {
                    auto ori = get_orientation<1, 2>(bd_key_2d);
                    auto e_idx_2d = get_index<1, 2>(bd_key_2d);
                    auto e_key_3d = edge_key_map.at(e_idx_2d);
                    face_3d.add_boundary_cell(e_key_3d, ori);
                }

                cc.add_cell<2>(face_3d);
            }

            return cc;
        } else {
            throw std::runtime_error("No supported elements found in MSH file. Only triangles "
                                     "(type 2) and tetrahedra (type 4) are supported.");
        }
    }
}

/// @brief Load a simplicial mesh from MSH format and convert to spacetime cell complex with uniform time sampling
/// @tparam DIM Dimension of output cell complex (3 for triangles→3D, 4 for tetrahedra→4D)
/// @param filename Path to the input MSH file
/// @param num_time_samples Number of uniform time samples per vertex
/// @param t_min Minimum time value
/// @param t_max Maximum time value
/// @param options Options controlling the loading behavior
/// @return CellComplex<DIM> containing the spacetime cell complex
///
/// @throws std::runtime_error if the file cannot be read or contains unsupported element types
///
/// Template specializations:
/// - DIM=3: Loads planar 2D triangle mesh and converts to 3D cell complex
///   - Expects triangle elements (MSH type 2)
///   - Drops z-coordinates from input vertices (uses only x, y)
///   - Time becomes the 3rd dimension (z-direction)
///   - Each triangle is extruded to create a triangular prism (3-cell)
///
/// - DIM=4: Loads 3D tetrahedral mesh and converts to 4D cell complex
///   - Expects tetrahedron elements (MSH type 4)
///   - Uses full 3D spatial vertices (x, y, z)
///   - Time becomes the 4th dimension
///   - Each tetrahedron is extruded to create a 4D prism (4-cell)
///
/// Usage:
/// @code
/// // Load a planar triangle mesh and create 3D columns
/// auto cc3d = load_msh_as_columns<3>("planar_mesh.msh", 10);
///
/// // Load a tetrahedral mesh and create 4D columns
/// auto cc4d = load_msh_as_columns<4>("tet_mesh.msh", 10);
///
/// // Custom time range
/// auto cc3d = load_msh_as_columns<3>("mesh.msh", 5, 0.0, 2.0);
/// @endcode
template <int DIM>
inline CellComplex<DIM> load_msh_as_columns(
    const std::string& filename,
    size_t num_time_samples,
    Scalar t_min,
    Scalar t_max,
    const LoadMshOptions& options)
{
    static_assert(DIM == 3 || DIM == 4, "load_msh_as_columns only supports DIM=3 or DIM=4");

    auto logger = get_logger();

    if (num_time_samples < 2) {
        logger->error("num_time_samples must be at least 2, got {}", num_time_samples);
        throw std::runtime_error("num_time_samples must be at least 2");
    }

    if (t_min >= t_max) {
        logger->error("t_min ({}) must be less than t_max ({})", t_min, t_max);
        throw std::runtime_error("t_min must be less than t_max");
    }

    // Load MSH file using mshio
    logger->debug("Loading MSH file for {}D simplicial columns: {}", DIM, filename);
    auto spec = mshio::load_msh(filename);

    if (options.verbose) {
        logger->info("Loading MSH file for {}D simplicial columns: {}", DIM, filename);
        logger->info("  Nodes: {}", spec.nodes.num_nodes);
        logger->info("  Elements: {}", spec.elements.num_elements);
        logger->info("  Time samples per vertex: {}", num_time_samples);
        logger->info("  Time range: [{}, {}]", t_min, t_max);
    }

    // Spatial dimension is DIM-1
    constexpr int SPATIAL_DIM = DIM - 1;

    // Extract vertices (SPATIAL_DIM coordinates, drop z for 3D case)
    // MSH 4.1 node tags are explicit and may be arbitrary or out-of-order, so
    // build a contiguous indexing for tags and place coordinates accordingly.
    ankerl::unordered_dense::map<size_t, size_t> node_tag_to_index;
    node_tag_to_index.reserve(spec.nodes.num_nodes);

    size_t next_vertex_index = 0;
    for (const auto& node_block : spec.nodes.entity_blocks) {
        if (node_block.tags.size() != node_block.num_nodes_in_block) {
            throw std::runtime_error("MSH node block tag count does not match node count");
        }

        for (size_t i = 0; i < node_block.num_nodes_in_block; ++i) {
            const auto [it, inserted] =
                node_tag_to_index.emplace(node_block.tags[i], next_vertex_index);
            if (!inserted) {
                throw std::runtime_error(
                    "MSH file contains duplicate node tag: " + std::to_string(node_block.tags[i]));
            }
            ++next_vertex_index;
        }
    }

    std::vector<Scalar> vertices(spec.nodes.num_nodes * SPATIAL_DIM);

    for (const auto& node_block : spec.nodes.entity_blocks) {
        size_t num_nodes = node_block.num_nodes_in_block;
        const auto& data = node_block.data;

        if (data.size() != num_nodes * 3) {
            throw std::runtime_error("MSH node block coordinate count does not match node count");
        }

        for (size_t i = 0; i < num_nodes; ++i) {
            const size_t vertex_index = node_tag_to_index.at(node_block.tags[i]);

            // MSH stores coordinates as (x, y, z)
            for (int d = 0; d < SPATIAL_DIM; ++d) {
                vertices[vertex_index * SPATIAL_DIM + d] = data[i * 3 + d];
            }
            // For DIM=3, we drop the z-coordinate (data[i * 3 + 2])
        }
    }

    // Extract simplices (triangles for DIM=3, tetrahedra for DIM=4)
    std::vector<size_t> simplices;

    // Element type: 2 for triangle, 4 for tetrahedron
    constexpr int element_type = (DIM == 3) ? 2 : 4;
    constexpr int vertices_per_element = (DIM == 3) ? 3 : 4;
    const char* element_name = (DIM == 3) ? "triangle" : "tetrahedron";
    const char* element_plural = (DIM == 3) ? "triangles" : "tetrahedra";

    for (const auto& element_block : spec.elements.entity_blocks) {
        if (element_block.element_type == element_type) {
            size_t num_elements = element_block.num_elements_in_block;
            const auto& data = element_block.data;

            // Each element is: [tag, v0, v1, v2, ...]
            for (size_t i = 0; i < num_elements; ++i) {
                size_t offset = i * (vertices_per_element + 1); // tag + vertices
                // Convert node tags to vertex indices
                for (int v = 0; v < vertices_per_element; ++v) {
                    simplices.push_back(node_tag_to_index.at(data[offset + 1 + v]));
                }
            }

            if (options.verbose) {
                logger->info("  Found {} {}", num_elements, element_plural);
            }
        } else if (options.verbose) {
            logger->warn(
                "  Skipping element type {} (not a {})",
                element_block.element_type,
                element_name);
        }
    }

    if (simplices.empty()) {
        std::string error_msg = "No ";
        error_msg += element_plural;
        error_msg += " found in MSH file. load_msh_as_columns<";
        error_msg += std::to_string(DIM);
        error_msg += "> requires a ";
        error_msg += element_name;
        error_msg += " mesh.";
        logger->error("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    // Create uniform time samples
    size_t num_vertices = vertices.size() / SPATIAL_DIM;
    std::vector<Scalar> time_samples;
    std::vector<size_t> time_start_indices;

    time_samples.reserve(num_vertices * num_time_samples);
    time_start_indices.reserve(num_vertices + 1);

    // Generate uniform time samples for each vertex
    for (size_t i = 0; i < num_vertices; ++i) {
        time_start_indices.push_back(i * num_time_samples);

        for (size_t t = 0; t < num_time_samples; ++t) {
            Scalar time = t_min + (t_max - t_min) * static_cast<Scalar>(t) /
                                      static_cast<Scalar>(num_time_samples - 1);
            time_samples.push_back(time);
        }
    }
    time_start_indices.push_back(num_vertices * num_time_samples);

    if (options.verbose) {
        logger->info("  Creating {}D cell complex with {} spatial vertices", DIM, num_vertices);
        logger->info("  Total spacetime vertices: {}", num_vertices * num_time_samples);
    }

    // Convert to cell complex using from_simplicial_columns
    logger->debug("Converting to {}D cell complex using simplicial columns", DIM);
    return from_simplicial_columns<DIM>(vertices, simplices, time_samples, time_start_indices);
}

// Convenience overloads with default parameters

/// @brief Load a simplicial mesh as columns with default time range [0, 1] and options
template <int DIM>
inline CellComplex<DIM> load_msh_as_columns(const std::string& filename, size_t num_time_samples)
{
    return load_msh_as_columns<DIM>(filename, num_time_samples, 0.0, 1.0, LoadMshOptions{});
}

/// @brief Load a simplicial mesh as columns with custom time range and default options
template <int DIM>
inline CellComplex<DIM> load_msh_as_columns(
    const std::string& filename,
    size_t num_time_samples,
    Scalar t_min,
    Scalar t_max)
{
    return load_msh_as_columns<DIM>(filename, num_time_samples, t_min, t_max, LoadMshOptions{});
}

// Note: This header is fully header-only. Templates are instantiated on-demand.
// No explicit instantiations needed.

} // namespace cell_complex
