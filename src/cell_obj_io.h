#pragma once

#include <cell_complex/Cell.h>
#include <cell_complex/CellComplex.h>
#include <cell_complex/triangulation.h>

#include <ankerl/unordered_dense.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cell_complex {

namespace detail {

/// @brief Collect ordered 1-based OBJ vertex indices for a polygonal 2-cell.
/// The boundary cycle is reconstructed from face.boundary; a Negative orientation
/// tag on face_key reverses the winding order.
template <int DIM>
std::vector<size_t> get_polygon_vertex_indices(
    const CellComplex<DIM>& cc,
    const typename Cell<2, DIM>::KeyType& face_key,
    const ankerl::unordered_dense::map<size_t, size_t>& vertex_to_idx)
{
    auto& face = cc.template get_cell<2>(face_key);
    auto ordered = build_ordered_boundary_cycle<DIM>(cc, face.boundary);

    std::vector<size_t> indices;
    indices.reserve(ordered.size());
    for (const auto& ek : ordered) {
        auto [tail, head] = get_edge_directed_endpoints<DIM>(cc, ek);
        auto it = vertex_to_idx.find(get_index<0, DIM>(tail));
        if (it == vertex_to_idx.end())
            throw std::runtime_error("Vertex not found in OBJ index map");
        indices.push_back(it->second);
    }

    if (get_orientation<2, DIM>(face_key) == Orientation::Negative)
        std::reverse(indices.begin(), indices.end());

    return indices;
}

} // namespace detail

/// @brief Save a cell complex to Wavefront OBJ format.
/// @param filename  Path to the output .obj file
/// @param cc        Cell complex to save (not modified)
///
/// Detects the highest-dimensional cells present:
/// - Highest dim 1: saves edges as \c l polyline elements
/// - Highest dim 2: saves polygonal 2-cells as \c f face elements (no triangulation)
/// - Highest dim 3+: saves boundary 2-cells of each 3-cell as \c f face elements
///
/// Vertex coordinates are written as \c "v x y z"; for DIM < 3 missing axes are
/// padded with 0, for DIM > 3 only the first 3 coordinates are written.
template <int DIM>
void save_obj(const std::string& filename, const CellComplex<DIM>& cc)
{
    static_assert(DIM >= 1, "save_obj requires at least 1D cell complexes");

    // Build vertex index map: slot_map key index -> 1-based OBJ index
    ankerl::unordered_dense::map<size_t, size_t> vertex_to_idx;
    auto& vertices = cc.template get_cells<0>();
    size_t vi = 1;
    for (const auto& [key, vref] : vertices.items()) {
        vertex_to_idx[get_index<0, DIM>(key)] = vi++;
    }

    std::ofstream out(filename);
    if (!out)
        throw std::runtime_error("Cannot open OBJ file for writing: " + filename);

    constexpr int spatial_dim = (DIM < 3) ? DIM : 3;
    for (const auto& [key, vref] : vertices.items()) {
        const auto& v = vref.get();
        out << "v";
        for (int d = 0; d < spatial_dim; ++d) out << ' ' << v.coordinates[d];
        for (int d = spatial_dim; d < 3; ++d) out << " 0";
        out << '\n';
    }

    const bool has_2_cells = [&]() {
        if constexpr (DIM >= 2) return cc.template num_cells<2>() > 0;
        return false;
    }();
    const bool has_3_cells = [&]() {
        if constexpr (DIM >= 3) return cc.template num_cells<3>() > 0;
        return false;
    }();

    if (!has_2_cells && !has_3_cells) {
        if constexpr (DIM >= 1) {
            auto& edges = cc.template get_cells<1>();
            for (auto& [edge_key, eref] : edges.items()) {
                auto oriented_key = set_orientation<1, DIM>(edge_key, Orientation::Positive);
                auto [tail, head] = detail::get_edge_directed_endpoints<DIM>(cc, oriented_key);
                out << "l "
                    << vertex_to_idx[get_index<0, DIM>(tail)] << ' '
                    << vertex_to_idx[get_index<0, DIM>(head)] << '\n';
            }
        }
    } else if (has_3_cells) {
        if constexpr (DIM >= 3) {
            auto& volumes = cc.template get_cells<3>();
            for (auto& [vol_key, vol_ref] : volumes.items()) {
                for (const auto& face_key : vol_ref.get().boundary) {
                    auto indices =
                        detail::get_polygon_vertex_indices<DIM>(cc, face_key, vertex_to_idx);
                    out << "f";
                    for (size_t idx : indices) out << ' ' << idx;
                    out << '\n';
                }
            }
        }
    } else {
        if constexpr (DIM >= 2) {
            auto& faces = cc.template get_cells<2>();
            for (auto& [face_key, fref] : faces.items()) {
                auto indices =
                    detail::get_polygon_vertex_indices<DIM>(cc, face_key, vertex_to_idx);
                out << "f";
                for (size_t idx : indices) out << ' ' << idx;
                out << '\n';
            }
        }
    }

    out.flush();
    if (!out)
        throw std::runtime_error("Failed writing OBJ file: " + filename);
}



template <int DIM>
void save_edges_to_obj(const std::string& path,
    std::vector<typename Cell<1, DIM>::KeyType >& junction_edges, 
    const CellComplex<DIM>& cc)
{
    // Save junction edges to OBJ
    std::ofstream junction_out(path);
    if (!junction_out) throw std::runtime_error("Cannot open junction_edges.obj");

    // Collect unique vertices referenced by selected edges, in insertion order
    ankerl::unordered_dense::map<size_t, size_t> vert_idx;  // raw index → 1-based OBJ index
    std::vector<typename Cell<0, DIM>::KeyType> vert_keys;

    for (const auto& ek : junction_edges) {
        for (const auto& bd : cc.template get_cell<1>(ek).boundary) {
            auto vk = strip_orientation<0, DIM>(bd);
            size_t raw = get_index<0, DIM>(vk);
            if (vert_idx.emplace(raw, vert_keys.size() + 1).second)
                vert_keys.push_back(vk);
        }
    }

    // Write vertices — OBJ only has x y z; for DIM=4 the 4th coordinate is time, drop it
    for (const auto& vk : vert_keys) {
        const auto& v = cc.template get_cell<0>(vk);
        junction_out << "v " << v.coordinates[0]
                    << " "  << v.coordinates[1]
                    << " "  << v.coordinates[2] << "\n";
    }

    // Write edges as polylines
    for (const auto& ek : junction_edges) {
        const auto& e = cc.template get_cell<1>(ek);
        junction_out << "l "
                    << vert_idx[cell_complex::get_index<0, 4>(e.boundary[0])]
                    << " "
                    << vert_idx[cell_complex::get_index<0, 4>(e.boundary[1])]
                    << "\n";
    }
}

template <int DIM>
void save_edges_to_ply(
    const std::string& path,
    const std::vector<typename Cell<1, DIM>::KeyType>& edges,
    const std::vector<int>& edge_labels,
    const CellComplex<DIM>& cc)
{
    if (edge_labels.size() != edges.size()) {
        throw std::runtime_error(
            "save_edges_to_ply: edge_labels.size() must equal edges.size()");
    }

    // Compact vertex set: only emit vertices referenced by these edges.
    ankerl::unordered_dense::map<size_t, size_t> vert_idx;  // raw idx -> 0-based PLY idx
    std::vector<typename Cell<0, DIM>::KeyType> vert_keys;
    std::vector<int> vert_labels;

     
    for (size_t i = 0; i <edges.size(); ++i) {
        const auto& ek = edges[i];
        for (const auto& bd : cc.template get_cell<1>(ek).boundary) {
            auto vk = strip_orientation<0, DIM>(bd);
            size_t raw = get_index<0, DIM>(vk);
            if (vert_idx.emplace(raw, vert_keys.size()).second) {
                vert_keys.push_back(vk);
                vert_labels.push_back(edge_labels[i]);
            }
        }
    }
    


    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open " + path + " for writing");

    // ASCII PLY header
    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "comment cell_complex junction edges with int labels\n";
    out << "element vertex " << vert_keys.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property float quality\n"; 
    if constexpr (DIM == 4) {
        out << "property float time\n";
    }
    out << "element edge " << edges.size() << "\n";
    out << "property int vertex1\n";
    out << "property int vertex2\n";
    out << "property int label\n";
    out << "end_header\n";

    // Vertex section
    for (size_t i = 0; i < vert_keys.size(); ++i) {
        const auto& vk = vert_keys[i];
        const auto& v = cc.template get_cell<0>(vk);
        out << v.coordinates[0] << ' '
            << v.coordinates[1] << ' '
            << v.coordinates[2] << ' '
            << static_cast<float>(vert_labels[i]);
        if constexpr (DIM == 4) {
            out << ' ' << v.coordinates[3];
        }
        out << '\n';
    }

    // Edge section
    for (size_t i = 0; i < edges.size(); ++i) {
        const auto& e = cc.template get_cell<1>(edges[i]);
        size_t a = vert_idx[get_index<0, DIM>(e.boundary[0])];
        size_t b = vert_idx[get_index<0, DIM>(e.boundary[1])];
        out << a << ' ' << b << ' ' << edge_labels[i] << '\n';
    }

    out.flush();
    if (!out) throw std::runtime_error("Failed writing " + path);
}

template <int DIM>
void save_edges_to_mathematica(
    const std::string& path,
    const std::vector<typename Cell<1, DIM>::KeyType>& edges,
    const std::vector<int>& edge_labels,
    const CellComplex<DIM>& cc)
{
    if (edge_labels.size() != edges.size()) {
        throw std::runtime_error(
            "save_edges_to_mathematica: edge_labels.size() must equal edges.size()");
    }

    // Compact vertex set: only emit vertices used by these edges.
    ankerl::unordered_dense::map<size_t, size_t> vert_idx;  // raw idx -> 1-based Mathematica idx
    std::vector<typename Cell<0, DIM>::KeyType> vert_keys;

    for (const auto& ek : edges) {
        for (const auto& bd : cc.template get_cell<1>(ek).boundary) {
            auto vk = strip_orientation<0, DIM>(bd);
            size_t raw = get_index<0, DIM>(vk);
            if (vert_idx.emplace(raw, vert_keys.size() + 1).second) {
                vert_keys.push_back(vk);
            }
        }
    }

    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open " + path + " for writing");

    out << std::setprecision(17);  // round-trip-safe doubles

    // ---------- header / association layout ----------
    out << "(* CellComplex junction edges with integer labels *)\n";
    out << "(* Use:                                                 *)\n";
    out << "(*   data = Get[\"" << path << "\"];                     *)\n";
    out << "(*   verts  = data[\"Vertices\"];                        *)\n";
    out << "(*   edges  = data[\"Edges\"];     (* {i,j} pairs *)     *)\n";
    out << "(*   labels = data[\"Labels\"];                          *)\n";
    out << "<|\n";

    // ---------- vertices ----------
    // Mathematica is 1-based; we already used 1-based indices above.
    // Emit each vertex as {x, y, z} or {x, y, z, t} for DIM == 4.
    out << "  \"Vertices\" -> {\n";
    for (size_t i = 0; i < vert_keys.size(); ++i) {
        const auto& v = cc.template get_cell<0>(vert_keys[i]);
        out << "    {"
            << v.coordinates[0] << ", "
            << v.coordinates[1] << ", "
            << v.coordinates[2];
        if constexpr (DIM == 4) {
            out << ", " << v.coordinates[3];
        }
        out << "}";
        if (i + 1 < vert_keys.size()) out << ",";
        out << "\n";
    }
    out << "  },\n";

    // ---------- edges ----------
    out << "  \"Edges\" -> {\n";
    for (size_t i = 0; i < edges.size(); ++i) {
        const auto& e = cc.template get_cell<1>(edges[i]);
        size_t a = vert_idx[get_index<0, DIM>(e.boundary[0])];
        size_t b = vert_idx[get_index<0, DIM>(e.boundary[1])];
        out << "    {" << a << ", " << b << "}";
        if (i + 1 < edges.size()) out << ",";
        out << "\n";
    }
    out << "  },\n";

    // ---------- labels ----------
    out << "  \"Labels\" -> {";
    for (size_t i = 0; i < edge_labels.size(); ++i) {
        out << edge_labels[i];
        if (i + 1 < edge_labels.size()) out << ", ";
    }
    out << "}\n";

    out << "|>\n";

    out.flush();
    if (!out) throw std::runtime_error("Failed writing " + path);
}

} // namespace cell_complex
