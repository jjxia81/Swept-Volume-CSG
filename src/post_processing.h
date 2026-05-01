//
//  post_processing.h
//  general_sweep
//
//  Created by Yiwen Ju on 5/13/25.
//

#ifndef post_processing_h
#define post_processing_h

#include <ankerl/unordered_dense.h>
#include <arrangement/Arrangement.h>
#include <lagrange/SurfaceMesh.h>
#include <lagrange/mesh_cleanup/remove_isolated_vertices.h>
#include <lagrange/utils/SmallVector.h>
#include <lagrange/views.h>

#include <algorithm>
#include "cell_msh_io.h"

template <typename Scalar, typename Index>
lagrange::SurfaceMesh<Scalar, Index> isocontour_to_mesh(mtetcol::Contour<4>& isocontour)
{
    lagrange::SurfaceMesh<Scalar, Index> envelope;

    // Port the isocontour into lagrange mesh
    size_t num_vertices = isocontour.get_num_vertices();
    size_t num_cycles = isocontour.get_num_cycles();

    // Add vertices and time
    envelope.add_vertices(num_vertices);
    envelope.template create_attribute<double>(
        "time",
        lagrange::AttributeElement::Vertex,
        lagrange::AttributeUsage::Scalar,
        1);
    auto time_values = attribute_vector_ref<double>(envelope, "time");

    for (size_t i = 0; i < num_vertices; i++) {
        auto xyzt = isocontour.get_vertex(i);
        auto pos = envelope.ref_position(i);
        pos[0] = xyzt[0];
        pos[1] = xyzt[1];
        pos[2] = xyzt[2];
        time_values[i] = xyzt[3];
    }

    // Add polygons
    lagrange::SmallVector<uint32_t, 16> polygon;
    for (size_t i = 0; i < num_cycles; i++) {
        auto cycle = isocontour.get_cycle(i);
        size_t cycle_size = cycle.size();
        polygon.clear();
        polygon.resize(cycle_size);

        size_t ind = 0;
        for (auto si : cycle) {
            mtetcol::Index seg_id = index(si);
            bool seg_ori = mtetcol::orientation(si);
            auto seg = isocontour.get_segment(seg_id);
            polygon[ind] = (seg_ori ? seg[0] : seg[1]);
            std::pair<mtetcol::Index, mtetcol::Index> edge_key = {
                std::min(seg[0], seg[1]),
                std::max(seg[0], seg[1])};

            ind++;
        }
        envelope.add_polygon({polygon.data(), polygon.size()});
    }

    // Add regular attribute
    envelope.template create_attribute<uint8_t>(
        "regular",
        lagrange::AttributeElement::Facet,
        lagrange::AttributeUsage::Scalar,
        1);
    auto regular_values = attribute_vector_ref<uint8_t>(envelope, "regular");
    for (size_t i = 0; i < num_cycles; i++) {
        regular_values[i] = isocontour.is_cycle_regular(i) ? 1 : 0;
    }

    return envelope;
}

template <typename Scalar, typename Index>
lagrange::SurfaceMesh<Scalar, Index> compute_envelope_arrangement(
    const lagrange::SurfaceMesh<Scalar, Index>& envelope, 
    Scalar vol_threshold,
    size_t face_count_threshold)
{
    using Point = Eigen::Matrix<Scalar, 3, 1>;

    // Compute arrangement
    auto V = vertex_view(envelope).template cast<double>();
    auto F = facet_view(envelope).template cast<int>();
    auto T = attribute_vector_view<Scalar>(envelope, "time");
    arrangement::VectorI face_labels = Eigen::VectorXi::LinSpaced(F.rows(), 0, F.rows() - 1);
    auto engine = arrangement::Arrangement::create_mesh_arrangement(V, F, face_labels);
    engine->run();

    lagrange::SurfaceMesh<Scalar, Index> sweep_arrangement;

    const auto& cell_data = engine->get_cells(); // (#facets x 2) array
    const auto& patches = engine->get_patches(); // list of facet indices
    const auto& parent_facets = engine->get_out_face_labels(); // size = #facets
    const auto& winding_number = engine->get_winding_number(); // (#facets x 2) array

    const auto& out_vertices = engine->get_vertices();
    const auto& arrangement_faces = engine->get_faces();
    size_t num_cells = engine->get_num_cells();
    size_t num_patches = engine->get_num_patches();
    size_t num_facets = arrangement_faces.rows();
    assert(patches.size() == num_facets);
    assert(cell_data.rows() == num_patches);

    sweep_arrangement.add_vertices(
        static_cast<Index>(out_vertices.rows()),
        {out_vertices.data(), static_cast<size_t>(out_vertices.size())});
    sweep_arrangement.add_triangles(static_cast<Index>(num_facets));
    {
        auto facets = facet_ref(sweep_arrangement);
        std::copy_n(arrangement_faces.data(), arrangement_faces.size(), facets.data());
    }
    sweep_arrangement.template create_attribute<Index>(
        "envelope_facet_id",
        lagrange::AttributeElement::Facet,
        lagrange::AttributeUsage::Scalar,
        1);
    auto envelope_facet_id = attribute_vector_ref<Index>(sweep_arrangement, "envelope_facet_id");
    std::copy_n(parent_facets.data(), parent_facets.size(), envelope_facet_id.data());
    sweep_arrangement.initialize_edges();

    // Build cell adjacency graph and compute cell volumes
    constexpr int32_t invalid_winding_number = std::numeric_limits<int32_t>::max();
    std::vector<ankerl::unordered_dense::set<Index>> cell_graph(num_cells);
    std::vector<Scalar> cell_volumes(num_cells, 0);
    std::vector<size_t> cell_face_counts(num_cells, 0);
    std::vector<int32_t> cell_winding_numbers(num_cells, invalid_winding_number);
    for (size_t fid = 0; fid < num_facets; fid++) {
        Index c0 = static_cast<Index>(cell_data(patches[fid], 0)); // Cell on the positive side
        Index c1 = static_cast<Index>(cell_data(patches[fid], 1)); // Cell on the negative side
        int w0 = winding_number(fid, 0); // Winding number on the positive side
        int w1 = winding_number(fid, 1); // Winding number on the negative side

        if (cell_winding_numbers[c0] == invalid_winding_number) {
            cell_winding_numbers[c0] = w0;
        } else {
            if (cell_winding_numbers[c0] != w0) {
                // This should never happen
                throw std::runtime_error("Inconsistent winding numbers detected!");
            }
        }
        if (cell_winding_numbers[c1] == invalid_winding_number) {
            cell_winding_numbers[c1] = w1;
        } else {
            if (cell_winding_numbers[c1] != w1) {
                // This should never happen
                throw std::runtime_error("Inconsistent winding numbers detected!");
            }
        }

        cell_graph[c0].insert(c1);
        cell_graph[c1].insert(c0);

        Eigen::Matrix<int, 1, 3> f = arrangement_faces.row(fid);
        Point p0 = out_vertices.row(f[0]).template cast<Scalar>();
        Point p1 = out_vertices.row(f[1]).template cast<Scalar>();
        Point p2 = out_vertices.row(f[2]).template cast<Scalar>();
        Scalar vol = p0.dot((p1).cross(p2));

        cell_volumes[c0] -= vol / 6;
        cell_volumes[c1] += vol / 6;
        cell_face_counts[c0]++;
        cell_face_counts[c1]++;
    }
    auto cell_is_small = [&](Index cid) {
        return (
            std::abs(cell_volumes[cid]) < vol_threshold ||
            cell_face_counts[cid] < face_count_threshold);
    };

    std::vector<int> parent_cell(num_cells);
    std::iota(parent_cell.begin(), parent_cell.end(), 0);
    auto get_parent = [&](auto&& self, int cid) -> int {
        if (parent_cell[cid] != cid) {
            parent_cell[cid] = self(self, parent_cell[cid]);
        }
        return parent_cell[cid];
    };
    for (size_t cid = 0; cid < num_cells; cid++) {
        if (cell_is_small(cid)) {
            // Union small cell with one of its neighbors
            int parent = parent_cell[cid];
            Scalar max_vol = std::abs(cell_volumes[cid]);
            for (auto adj_cid : cell_graph[cid]) {
                if (std::abs(cell_volumes[adj_cid]) > max_vol) {
                    max_vol = std::abs(cell_volumes[adj_cid]);
                    parent = adj_cid;
                }
            }
            parent_cell[cid] = get_parent(get_parent, parent);
        }
    }

    // Compute sweep surface facets
    sweep_arrangement.template create_attribute<int8_t>(
        "valid",
        lagrange::AttributeElement::Facet,
        lagrange::AttributeUsage::Scalar,
        1);
    auto is_valid = attribute_vector_ref<int8_t>(sweep_arrangement, "valid");
    is_valid.setZero();
    size_t num_valid_facets = 0;
    for (size_t fid = 0; fid < num_facets; fid++) {
        int c0 = cell_data(patches[fid], 0); // Cell on the positive side
        int c1 = cell_data(patches[fid], 1); // Cell on the negative side
        c0 = get_parent(get_parent, c0); // Find the representative parent cell
        c1 = get_parent(get_parent, c1); // Find the representative parent cell
        int w0 = cell_winding_numbers[c0]; // Winding number on the positive side
        int w1 = cell_winding_numbers[c1]; // Winding number on the negative side

        if (w0 == 0 && w1 != 0) {
            is_valid[fid] = 1;
            num_valid_facets++;
        } else if (w1 == 0 && w0 != 0) {
            is_valid[fid] = -1;
            num_valid_facets++;
        }
    }

    // Compute per-corner time attribute
    auto interpolate_time = [&](Index fid, Point p) {
        Eigen::Matrix<int, 1, 3> f = F.row(fid);
        Point p0 = V.row(f[0]);
        Point p1 = V.row(f[1]);
        Point p2 = V.row(f[2]);
        auto t0 = T[f[0]];
        auto t1 = T[f[1]];
        auto t2 = T[f[2]];

        Scalar b0 = ((p1 - p).cross(p2 - p)).norm();
        Scalar b1 = ((p2 - p).cross(p0 - p)).norm();
        Scalar b2 = ((p0 - p).cross(p1 - p)).norm();
        Scalar denom = b0 + b1 + b2;
        if (std::abs(denom) > std::numeric_limits<Scalar>::epsilon()) {
            return (t0 * b0 + t1 * b1 + t2 * b2) / denom;
        } else {
            // Simple average to avoid numerical issues
            return (t0 + t1 + t2) / Scalar(3);
        }
    };

    sweep_arrangement.template create_attribute<Scalar>(
        "time",
        lagrange::AttributeElement::Corner,
        lagrange::AttributeUsage::Scalar,
        1);
    auto time_values = attribute_vector_ref<Scalar>(sweep_arrangement, "time");
    Index count = 0;
    // TODO: parallelize this loop
    for (size_t fid = 0; fid < num_facets; fid++) {
        auto idx = count++;
        auto parent_fid = parent_facets(fid);
        Index v0 = static_cast<Index>(arrangement_faces(fid, 0));
        Index v1 = static_cast<Index>(arrangement_faces(fid, 1));
        Index v2 = static_cast<Index>(arrangement_faces(fid, 2));
        Point p0(out_vertices.row(v0).template cast<Scalar>());
        Point p1(out_vertices.row(v1).template cast<Scalar>());
        Point p2(out_vertices.row(v2).template cast<Scalar>());
        Scalar t0 = interpolate_time(parent_fid, p0);
        Scalar t1 = interpolate_time(parent_fid, p1);
        Scalar t2 = interpolate_time(parent_fid, p2);
        time_values[3 * idx + 0] = t0;
        time_values[3 * idx + 1] = t1;
        time_values[3 * idx + 2] = t2;
    }

    // Extract feature edges
    sweep_arrangement.template create_attribute<int8_t>(
        "is_feature",
        lagrange::AttributeElement::Edge,
        lagrange::AttributeUsage::Scalar,
        1);
    auto is_feature = attribute_vector_ref<int8_t>(sweep_arrangement, "is_feature");
    is_feature.setZero();
    Index num_edges = sweep_arrangement.get_num_edges();
    for (Index eid = 0; eid < num_edges; eid++) {
        Index edge_valence = sweep_arrangement.count_num_corners_around_edge(eid);
        if (edge_valence != 2) {
            bool feature_edge = false;
            sweep_arrangement.foreach_facet_around_edge(eid, [&](Index fid) {
                if (is_valid[fid] != 0) {
                    feature_edge = true;
                }
            });
            if (feature_edge) is_feature[eid] = 1;
        }
    }

    return sweep_arrangement;
}

template <typename Scalar, typename Index>
lagrange::SurfaceMesh<Scalar, Index> extract_sweep_surface_from_arrangement(
    lagrange::SurfaceMesh<Scalar, Index>& sweep_arrangement)
{
    Index num_arrangement_facets = sweep_arrangement.get_num_facets();
    auto V = vertex_view(sweep_arrangement);
    auto F = facet_view(sweep_arrangement);
    auto is_valid = attribute_vector_view<int8_t>(sweep_arrangement, "valid");
    auto time_values = attribute_vector_view<Scalar>(sweep_arrangement, "time");

    lagrange::SurfaceMesh<Scalar, Index> sweep_surface;
    sweep_surface.add_vertices(
        static_cast<Index>(V.rows()),
        {V.data(), static_cast<size_t>(V.size())});

    Index num_valid_facets = 0;
    for (Index fid = 0; fid < num_arrangement_facets; fid++) {
        if (is_valid[fid] != 0) num_valid_facets++;
    }
    sweep_surface.add_triangles(num_valid_facets);
    auto sweep_F = facet_ref(sweep_surface);

    Index count = 0;
    for (Index fid = 0; fid < num_arrangement_facets; fid++) {
        if (is_valid[fid] == 0) {
            continue;
        } else if (is_valid[fid] == 1) {
            sweep_F.row(count) = F.row(fid);
        } else {
            sweep_F.row(count) = F.row(fid).reverse();
        }
        count++;
    }

    sweep_surface.template create_attribute<Scalar>(
        "time",
        lagrange::AttributeElement::Corner,
        lagrange::AttributeUsage::Scalar,
        1);
    auto sweep_time_values = attribute_vector_ref<Scalar>(sweep_surface, "time");

    count = 0;
    for (Index fid = 0; fid < num_arrangement_facets; fid++) {
        if (is_valid[fid] == 0) {
            continue;
        }
        const Index sweep_c_begin = sweep_surface.get_facet_corner_begin(count);
        const Index arrang_c_begin = sweep_arrangement.get_facet_corner_begin(fid);
        if (is_valid[fid] == 1) {
            sweep_time_values[sweep_c_begin + 0] = time_values[arrang_c_begin + 0];
            sweep_time_values[sweep_c_begin + 1] = time_values[arrang_c_begin + 1];
            sweep_time_values[sweep_c_begin + 2] = time_values[arrang_c_begin + 2];
        } else {
            sweep_time_values[sweep_c_begin + 0] = time_values[arrang_c_begin + 2];
            sweep_time_values[sweep_c_begin + 1] = time_values[arrang_c_begin + 1];
            sweep_time_values[sweep_c_begin + 2] = time_values[arrang_c_begin + 0];
        }
        count++;
    }

    lagrange::remove_isolated_vertices(sweep_surface);
    return sweep_surface;
}

template <typename Scalar, typename Index>
lagrange::SurfaceMesh<Scalar, Index> envelope_complex_to_mesh(
    const cell_complex::CellComplex<4>& cc)
{
    using namespace cell_complex;
    lagrange::SurfaceMesh<Scalar, Index> mesh;

    // --- vertices: slot key index -> 0-based mesh index ---
    auto& vertices = cc.template get_cells<0>();
    const Index num_vertices = static_cast<Index>(vertices.size());
    mesh.add_vertices(num_vertices);

    mesh.template create_attribute<Scalar>(
        "time",
        lagrange::AttributeElement::Vertex,
        lagrange::AttributeUsage::Scalar,
        1);
    auto time_values = lagrange::attribute_vector_ref<Scalar>(mesh, "time");

    ankerl::unordered_dense::map<size_t, Index> vertex_to_idx;
    vertex_to_idx.reserve(num_vertices);

    Index vi = 0;
    for (const auto& [key, vref] : vertices.items()) {
        const auto& v = vref.get();
        vertex_to_idx[get_index<0, 4>(key)] = vi;
        auto pos = mesh.ref_position(vi);
        pos[0] = static_cast<Scalar>(v.coordinates[0]);
        pos[1] = static_cast<Scalar>(v.coordinates[1]);
        pos[2] = static_cast<Scalar>(v.coordinates[2]);
        time_values[vi] = static_cast<Scalar>(v.coordinates[3]);
        ++vi;
    }

    // --- facets: walk each 2-cell's boundary cycle directly ---
    auto& faces = cc.template get_cells<2>();
    lagrange::SmallVector<Index, 16> polygon;

    for (auto& [face_key, fref] : faces.items()) {
        const auto& face = fref.get();

        // Reconstruct ordered boundary cycle of edges
        auto ordered = detail::build_ordered_boundary_cycle<4>(cc, face.boundary);

        polygon.clear();
        polygon.reserve(ordered.size());
        for (const auto& edge_key : ordered) {
            auto [tail, head] = detail::get_edge_directed_endpoints<4>(cc, edge_key);
            auto it = vertex_to_idx.find(get_index<0, 4>(tail));
            if (it == vertex_to_idx.end())
                throw std::runtime_error("Vertex not found in envelope mesh map");
            polygon.push_back(it->second);
        }

        // Apply 2-cell's stored orientation: Negative reverses winding
        auto face_key_pos = set_orientation<2, 4>(face_key, Orientation::Positive);
        // (iteration keys typically have Unknown orientation; we want the *intrinsic* one,
        //  which the cycle already encodes — so just check the key as iterated.)
        if (get_orientation<2, 4>(face_key) == Orientation::Negative) {
            std::reverse(polygon.begin(), polygon.end());
        }
        (void)face_key_pos;

        mesh.add_polygon({polygon.data(), polygon.size()});
    }

    return mesh;
}

template <typename Scalar, typename Index>
lagrange::SurfaceMesh<Scalar, Index> envelope_complex_to_mesh(
    const cell_complex::CellComplex<4>& cc,
    size_t root_start,
    size_t num_leafs,
    const ankerl::unordered_dense::set<size_t>& junction_edge_raw_ids)
{
    using namespace cell_complex;
    lagrange::SurfaceMesh<Scalar, Index> mesh;

    // --- vertices (unchanged) ---
    auto& vertices = cc.template get_cells<0>();
    const Index num_vertices = static_cast<Index>(vertices.size());
    mesh.add_vertices(num_vertices);

    mesh.template create_attribute<Scalar>(
        "time", lagrange::AttributeElement::Vertex,
        lagrange::AttributeUsage::Scalar, 1);
    auto time_values = lagrange::attribute_vector_ref<Scalar>(mesh, "time");

    ankerl::unordered_dense::map<size_t, Index> vertex_to_idx;
    vertex_to_idx.reserve(num_vertices);

    Index vi = 0;
    for (const auto& [key, vref] : vertices.items()) {
        const auto& v = vref.get();
        vertex_to_idx[get_index<0, 4>(key)] = vi;
        auto pos = mesh.ref_position(vi);
        pos[0] = static_cast<Scalar>(v.coordinates[0]);
        pos[1] = static_cast<Scalar>(v.coordinates[1]);
        pos[2] = static_cast<Scalar>(v.coordinates[2]);
        time_values[vi] = static_cast<Scalar>(v.coordinates[3]);
        ++vi;
    }

    // --- facets + per-facet `face_label`, `face_dom_chunk`
    //     + per-corner `corner_is_junction` ---
    auto& faces = cc.template get_cells<2>();

    // Pre-count facets so we can size attributes after add_polygon calls.
    // We'll create attributes after building topology to keep storage stable.
    lagrange::SmallVector<Index, 16> polygon;

    // Collect per-polygon data while building, write into attributes after.
    std::vector<uint8_t> facet_labels;
    std::vector<uint64_t> facet_dom_chunks;  // pack chunk to a uint64 (works for num_leafs <= 64)
    std::vector<uint8_t> corner_is_junction; // flat per-corner

    facet_labels.reserve(faces.size());
    facet_dom_chunks.reserve(faces.size());
    corner_is_junction.reserve(faces.size() * 3);

    for (auto& [face_key, fref] : faces.items()) {
        const auto& face = fref.get();

        auto ordered = detail::build_ordered_boundary_cycle<4>(cc, face.boundary);

        polygon.clear();
        polygon.reserve(ordered.size());
        // Track per-edge junction flag in cycle order.
        lagrange::SmallVector<uint8_t, 16> edge_is_junction;
        edge_is_junction.reserve(ordered.size());

        for (const auto& edge_key : ordered) {
            auto [tail, head] = detail::get_edge_directed_endpoints<4>(cc, edge_key);
            auto it = vertex_to_idx.find(get_index<0, 4>(tail));
            if (it == vertex_to_idx.end())
                throw std::runtime_error("Vertex not found in envelope mesh map");
            polygon.push_back(it->second);

            size_t edge_raw = get_index<1, 4>(edge_key);
            edge_is_junction.push_back(
                junction_edge_raw_ids.contains(edge_raw) ? 1 : 0);
        }

        // Negative orientation: reverse winding AND the edge flags
        // (corner i of a triangle owns the edge from corner i to corner i+1,
        //  so reversing the vertex order also reverses which edge each corner owns).
        if (get_orientation<2, 4>(face_key) == Orientation::Negative) {
            std::reverse(polygon.begin(), polygon.end());
            // After reversing, corner i now owns what used to be edge (n-1-i, n-2-i):
            //   new corner 0 -> edge between old vert (n-1) and old vert (n-2)  -> old edge n-2
            //   new corner i -> old edge (n-2-i)
            // Easier: reverse and rotate.
            std::reverse(edge_is_junction.begin(), edge_is_junction.end());
            // After the reverse, edge_is_junction[0] currently holds the flag for
            // old corner (n-1)'s outgoing edge, which goes from old vert (n-1) to
            // old vert (0) — but new corner 0 owns the edge from old vert (n-1)
            // to old vert (n-2), which was old corner (n-2)'s edge. Rotate by 1
            // to fix.
            std::rotate(
                edge_is_junction.begin(),
                edge_is_junction.begin() + 1,
                edge_is_junction.end());
        }

        mesh.add_polygon({polygon.data(), polygon.size()});

        facet_labels.push_back(face.label);
        auto chunk = get_chunk(face.dominance, root_start, num_leafs);
        facet_dom_chunks.push_back(chunk.to_ullong());

        for (uint8_t flag : edge_is_junction) {
            corner_is_junction.push_back(flag);
        }
    }

    // Now write attributes
    mesh.template create_attribute<uint8_t>(
        "face_label", lagrange::AttributeElement::Facet,
        lagrange::AttributeUsage::Scalar, 1);
    auto face_label_attr = lagrange::attribute_vector_ref<uint8_t>(mesh, "face_label");
    std::copy(facet_labels.begin(), facet_labels.end(), face_label_attr.data());

    mesh.template create_attribute<uint64_t>(
        "face_dom_chunk", lagrange::AttributeElement::Facet,
        lagrange::AttributeUsage::Scalar, 1);
    auto face_dom_attr = lagrange::attribute_vector_ref<uint64_t>(mesh, "face_dom_chunk");
    std::copy(facet_dom_chunks.begin(), facet_dom_chunks.end(), face_dom_attr.data());

    mesh.template create_attribute<uint8_t>(
        "corner_is_junction", lagrange::AttributeElement::Corner,
        lagrange::AttributeUsage::Scalar, 1);
    auto corner_attr = lagrange::attribute_vector_ref<uint8_t>(mesh, "corner_is_junction");
    std::copy(corner_is_junction.begin(), corner_is_junction.end(), corner_attr.data());

    return mesh;
}
template <typename Scalar, typename Index>
lagrange::SurfaceMesh<Scalar, Index> envelope_complex_to_mesh2(
    const cell_complex::CellComplex<4>& cc,
    size_t root_start,
    size_t num_leafs,
    const ankerl::unordered_dense::map<size_t, int>& junction_edge_labels)
{
    using namespace cell_complex;
    lagrange::SurfaceMesh<Scalar, Index> mesh;

    // --- vertices ---
    auto& vertices = cc.template get_cells<0>();
    const Index num_vertices = static_cast<Index>(vertices.size());
    mesh.add_vertices(num_vertices);

    mesh.template create_attribute<Scalar>(
        "time", lagrange::AttributeElement::Vertex,
        lagrange::AttributeUsage::Scalar, 1);
    auto time_values = lagrange::attribute_vector_ref<Scalar>(mesh, "time");

    ankerl::unordered_dense::map<size_t, Index> vertex_to_idx;
    vertex_to_idx.reserve(num_vertices);

    Index vi = 0;
    for (const auto& [key, vref] : vertices.items()) {
        const auto& v = vref.get();
        vertex_to_idx[get_index<0, 4>(key)] = vi;
        auto pos = mesh.ref_position(vi);
        pos[0] = static_cast<Scalar>(v.coordinates[0]);
        pos[1] = static_cast<Scalar>(v.coordinates[1]);
        pos[2] = static_cast<Scalar>(v.coordinates[2]);
        time_values[vi] = static_cast<Scalar>(v.coordinates[3]);
        ++vi;
    }

    // --- facets + per-facet/per-corner attributes ---
    auto& faces = cc.template get_cells<2>();

    lagrange::SmallVector<Index, 16> polygon;

    std::vector<uint8_t> facet_labels;
    std::vector<uint64_t> facet_dom_chunks;
    std::vector<int32_t> corner_edge_label;  // per-corner int label (0 = none)

    facet_labels.reserve(faces.size());
    facet_dom_chunks.reserve(faces.size());
    corner_edge_label.reserve(faces.size() * 3);

    for (auto& [face_key, fref] : faces.items()) {
        const auto& face = fref.get();

        auto ordered = detail::build_ordered_boundary_cycle<4>(cc, face.boundary);

        polygon.clear();
        polygon.reserve(ordered.size());
        lagrange::SmallVector<int32_t, 16> edge_labels_cycle;
        edge_labels_cycle.reserve(ordered.size());

        for (const auto& edge_key : ordered) {
            auto [tail, head] = detail::get_edge_directed_endpoints<4>(cc, edge_key);
            auto it = vertex_to_idx.find(get_index<0, 4>(tail));
            if (it == vertex_to_idx.end())
                throw std::runtime_error("Vertex not found in envelope mesh map");
            polygon.push_back(it->second);

            size_t edge_raw = get_index<1, 4>(edge_key);
            auto lit = junction_edge_labels.find(edge_raw);
            edge_labels_cycle.push_back(
                lit != junction_edge_labels.end() ? lit->second : 0);
        }

        // Negative orientation: same reverse-then-rotate as before.
        if (get_orientation<2, 4>(face_key) == Orientation::Negative) {
            std::reverse(polygon.begin(), polygon.end());
            std::reverse(edge_labels_cycle.begin(), edge_labels_cycle.end());
            std::rotate(
                edge_labels_cycle.begin(),
                edge_labels_cycle.begin() + 1,
                edge_labels_cycle.end());
        }

        mesh.add_polygon({polygon.data(), polygon.size()});

        facet_labels.push_back(face.label);
        auto chunk = get_chunk(face.dominance, root_start, num_leafs);
        facet_dom_chunks.push_back(chunk.to_ullong());

        for (int32_t lbl : edge_labels_cycle) {
            corner_edge_label.push_back(lbl);
        }
    }

    mesh.template create_attribute<uint8_t>(
        "face_label", lagrange::AttributeElement::Facet,
        lagrange::AttributeUsage::Scalar, 1);
    auto face_label_attr = lagrange::attribute_vector_ref<uint8_t>(mesh, "face_label");
    std::copy(facet_labels.begin(), facet_labels.end(), face_label_attr.data());

    mesh.template create_attribute<uint64_t>(
        "face_dom_chunk", lagrange::AttributeElement::Facet,
        lagrange::AttributeUsage::Scalar, 1);
    auto face_dom_attr = lagrange::attribute_vector_ref<uint64_t>(mesh, "face_dom_chunk");
    std::copy(facet_dom_chunks.begin(), facet_dom_chunks.end(), face_dom_attr.data());

    mesh.template create_attribute<int32_t>(
        "corner_edge_label", lagrange::AttributeElement::Corner,
        lagrange::AttributeUsage::Scalar, 1);
    auto corner_attr = lagrange::attribute_vector_ref<int32_t>(mesh, "corner_edge_label");
    std::copy(corner_edge_label.begin(), corner_edge_label.end(), corner_attr.data());

    return mesh;
}

template <typename Scalar, typename Index>
lagrange::SurfaceMesh<Scalar, Index> compute_envelope_arrangement2(
    const lagrange::SurfaceMesh<Scalar, Index>& envelope,
    Scalar vol_threshold,
    size_t face_count_threshold)
{
    using Point = Eigen::Matrix<Scalar, 3, 1>;

    // Compute arrangement
    auto V = vertex_view(envelope).template cast<double>();
    auto F = facet_view(envelope).template cast<int>();
    auto T = attribute_vector_view<Scalar>(envelope, "time");
    arrangement::VectorI face_labels = Eigen::VectorXi::LinSpaced(F.rows(), 0, F.rows() - 1);
    auto engine = arrangement::Arrangement::create_mesh_arrangement(V, F, face_labels);
    engine->run();

    lagrange::SurfaceMesh<Scalar, Index> sweep_arrangement;

    const auto& cell_data = engine->get_cells();              // (#patches x 2)
    const auto& patches = engine->get_patches();              // size = #facets
    const auto& parent_facets = engine->get_out_face_labels();// size = #facets
    const auto& winding_number = engine->get_winding_number();// (#facets x 2)

    const auto& out_vertices = engine->get_vertices();
    const auto& arrangement_faces = engine->get_faces();
    size_t num_cells = engine->get_num_cells();
    size_t num_patches = engine->get_num_patches();
    size_t num_facets = arrangement_faces.rows();
    assert(patches.size() == num_facets);
    assert(cell_data.rows() == num_patches);
    (void)num_patches;

    sweep_arrangement.add_vertices(
        static_cast<Index>(out_vertices.rows()),
        {out_vertices.data(), static_cast<size_t>(out_vertices.size())});
    sweep_arrangement.add_triangles(static_cast<Index>(num_facets));
    {
        auto facets = facet_ref(sweep_arrangement);
        std::copy_n(arrangement_faces.data(), arrangement_faces.size(), facets.data());
    }
    sweep_arrangement.template create_attribute<Index>(
        "envelope_facet_id",
        lagrange::AttributeElement::Facet,
        lagrange::AttributeUsage::Scalar,
        1);
    auto envelope_facet_id = attribute_vector_ref<Index>(sweep_arrangement, "envelope_facet_id");
    std::copy_n(parent_facets.data(), parent_facets.size(), envelope_facet_id.data());
    sweep_arrangement.initialize_edges();

    // ============================================================
    // Propagate per-facet attributes from input envelope
    //   face_label, face_dom_chunk
    // ============================================================
    const bool has_face_label = envelope.has_attribute("face_label");
    const bool has_face_dom = envelope.has_attribute("face_dom_chunk");

    if (has_face_label) {
        auto in_face_label = attribute_vector_view<uint8_t>(envelope, "face_label");
        sweep_arrangement.template create_attribute<uint8_t>(
            "face_label",
            lagrange::AttributeElement::Facet,
            lagrange::AttributeUsage::Scalar,
            1);
        auto out_face_label = attribute_vector_ref<uint8_t>(sweep_arrangement, "face_label");
        for (size_t fid = 0; fid < num_facets; ++fid) {
            out_face_label[fid] = in_face_label[parent_facets(fid)];
        }
    }
    if (has_face_dom) {
        auto in_face_dom = attribute_vector_view<uint64_t>(envelope, "face_dom_chunk");
        sweep_arrangement.template create_attribute<uint64_t>(
            "face_dom_chunk",
            lagrange::AttributeElement::Facet,
            lagrange::AttributeUsage::Scalar,
            1);
        auto out_face_dom = attribute_vector_ref<uint64_t>(sweep_arrangement, "face_dom_chunk");
        for (size_t fid = 0; fid < num_facets; ++fid) {
            out_face_dom[fid] = in_face_dom[parent_facets(fid)];
        }
    }

    // Build cell adjacency graph and compute cell volumes
    constexpr int32_t invalid_winding_number = std::numeric_limits<int32_t>::max();
    std::vector<ankerl::unordered_dense::set<Index>> cell_graph(num_cells);
    std::vector<Scalar> cell_volumes(num_cells, 0);
    std::vector<size_t> cell_face_counts(num_cells, 0);
    std::vector<int32_t> cell_winding_numbers(num_cells, invalid_winding_number);
    for (size_t fid = 0; fid < num_facets; fid++) {
        Index c0 = static_cast<Index>(cell_data(patches[fid], 0));
        Index c1 = static_cast<Index>(cell_data(patches[fid], 1));
        int w0 = winding_number(fid, 0);
        int w1 = winding_number(fid, 1);

        if (cell_winding_numbers[c0] == invalid_winding_number) {
            cell_winding_numbers[c0] = w0;
        } else if (cell_winding_numbers[c0] != w0) {
            throw std::runtime_error("Inconsistent winding numbers detected!");
        }
        if (cell_winding_numbers[c1] == invalid_winding_number) {
            cell_winding_numbers[c1] = w1;
        } else if (cell_winding_numbers[c1] != w1) {
            throw std::runtime_error("Inconsistent winding numbers detected!");
        }

        cell_graph[c0].insert(c1);
        cell_graph[c1].insert(c0);

        Eigen::Matrix<int, 1, 3> f = arrangement_faces.row(fid);
        Point p0 = out_vertices.row(f[0]).template cast<Scalar>();
        Point p1 = out_vertices.row(f[1]).template cast<Scalar>();
        Point p2 = out_vertices.row(f[2]).template cast<Scalar>();
        Scalar vol = p0.dot((p1).cross(p2));

        cell_volumes[c0] -= vol / 6;
        cell_volumes[c1] += vol / 6;
        cell_face_counts[c0]++;
        cell_face_counts[c1]++;
    }
    auto cell_is_small = [&](Index cid) {
        return (
            std::abs(cell_volumes[cid]) < vol_threshold ||
            cell_face_counts[cid] < face_count_threshold);
    };

    std::vector<int> parent_cell(num_cells);
    std::iota(parent_cell.begin(), parent_cell.end(), 0);
    auto get_parent = [&](auto&& self, int cid) -> int {
        if (parent_cell[cid] != cid) {
            parent_cell[cid] = self(self, parent_cell[cid]);
        }
        return parent_cell[cid];
    };
    for (size_t cid = 0; cid < num_cells; cid++) {
        if (cell_is_small(cid)) {
            int parent = parent_cell[cid];
            Scalar max_vol = std::abs(cell_volumes[cid]);
            for (auto adj_cid : cell_graph[cid]) {
                if (std::abs(cell_volumes[adj_cid]) > max_vol) {
                    max_vol = std::abs(cell_volumes[adj_cid]);
                    parent = adj_cid;
                }
            }
            parent_cell[cid] = get_parent(get_parent, parent);
        }
    }

    // Compute sweep surface facets
    sweep_arrangement.template create_attribute<int8_t>(
        "valid",
        lagrange::AttributeElement::Facet,
        lagrange::AttributeUsage::Scalar,
        1);
    auto is_valid = attribute_vector_ref<int8_t>(sweep_arrangement, "valid");
    is_valid.setZero();
    size_t num_valid_facets = 0;
    for (size_t fid = 0; fid < num_facets; fid++) {
        int c0 = cell_data(patches[fid], 0);
        int c1 = cell_data(patches[fid], 1);
        c0 = get_parent(get_parent, c0);
        c1 = get_parent(get_parent, c1);
        int w0 = cell_winding_numbers[c0];
        int w1 = cell_winding_numbers[c1];

        if (w0 == 0 && w1 != 0) {
            is_valid[fid] = 1;
            num_valid_facets++;
        } else if (w1 == 0 && w0 != 0) {
            is_valid[fid] = -1;
            num_valid_facets++;
        }
    }
    (void)num_valid_facets;

    // Compute per-corner time attribute
    auto interpolate_time = [&](Index fid, Point p) {
        Eigen::Matrix<int, 1, 3> f = F.row(fid);
        Point p0 = V.row(f[0]);
        Point p1 = V.row(f[1]);
        Point p2 = V.row(f[2]);
        auto t0 = T[f[0]];
        auto t1 = T[f[1]];
        auto t2 = T[f[2]];

        Scalar b0 = ((p1 - p).cross(p2 - p)).norm();
        Scalar b1 = ((p2 - p).cross(p0 - p)).norm();
        Scalar b2 = ((p0 - p).cross(p1 - p)).norm();
        Scalar denom = b0 + b1 + b2;
        if (std::abs(denom) > std::numeric_limits<Scalar>::epsilon()) {
            return (t0 * b0 + t1 * b1 + t2 * b2) / denom;
        } else {
            return (t0 + t1 + t2) / Scalar(3);
        }
    };

    sweep_arrangement.template create_attribute<Scalar>(
        "time",
        lagrange::AttributeElement::Corner,
        lagrange::AttributeUsage::Scalar,
        1);
    auto time_values = attribute_vector_ref<Scalar>(sweep_arrangement, "time");
    Index count = 0;
    for (size_t fid = 0; fid < num_facets; fid++) {
        auto idx = count++;
        auto parent_fid = parent_facets(fid);
        Index v0 = static_cast<Index>(arrangement_faces(fid, 0));
        Index v1 = static_cast<Index>(arrangement_faces(fid, 1));
        Index v2 = static_cast<Index>(arrangement_faces(fid, 2));
        Point p0(out_vertices.row(v0).template cast<Scalar>());
        Point p1(out_vertices.row(v1).template cast<Scalar>());
        Point p2(out_vertices.row(v2).template cast<Scalar>());
        Scalar t0 = interpolate_time(parent_fid, p0);
        Scalar t1 = interpolate_time(parent_fid, p1);
        Scalar t2 = interpolate_time(parent_fid, p2);
        time_values[3 * idx + 0] = t0;
        time_values[3 * idx + 1] = t1;
        time_values[3 * idx + 2] = t2;
    }

    // // ============================================================
    // // Recover per-edge `is_junction` on the arrangement
    // //   For each output edge eid, walk incident output facets, look up
    // //   their parent input facet, and test whether eid's geometry is
    // //   collinear with any junction-tagged corner-edge of that parent.
    // // ============================================================
    // const bool has_corner_junction = envelope.has_attribute("corner_is_junction");
    // sweep_arrangement.template create_attribute<int8_t>(
    //     "is_junction",
    //     lagrange::AttributeElement::Edge,
    //     lagrange::AttributeUsage::Scalar,
    //     1);
    // auto is_junction = attribute_vector_ref<int8_t>(sweep_arrangement, "is_junction");
    // is_junction.setZero();

    // if (has_corner_junction) {
    //     auto in_corner_junction =
    //         attribute_vector_view<uint8_t>(envelope, "corner_is_junction");
    //     auto in_V = vertex_view(envelope).template cast<Scalar>();

    //     const Scalar tol = Scalar(1e-9);

    //     // Predicate: is point q on segment [a,b] (collinear, parameter in [0,1])?
    //     auto on_segment = [&](const Point& a, const Point& b, const Point& q) {
    //         Point ab = b - a;
    //         Point aq = q - a;
    //         Scalar ab_len2 = ab.squaredNorm();
    //         if (ab_len2 < std::numeric_limits<Scalar>::epsilon() * std::numeric_limits<Scalar>::epsilon()) {
    //             return aq.squaredNorm() < tol * tol;
    //         }
    //         // Distance of q from infinite line through a,b:
    //         Scalar cross_norm = ab.cross(aq).norm();
    //         if (cross_norm > tol * std::sqrt(ab_len2)) return false;
    //         Scalar t = ab.dot(aq) / ab_len2;
    //         return t >= -tol && t <= Scalar(1) + tol;
    //     };

    //     const Index num_arr_edges = sweep_arrangement.get_num_edges();
    //     for (Index eid = 0; eid < num_arr_edges; ++eid) {
    //         // Endpoint coordinates of this output edge
    //         auto [v0, v1] = sweep_arrangement.get_edge_vertices(eid);
    //         Point q0 = out_vertices.row(v0).template cast<Scalar>();
    //         Point q1 = out_vertices.row(v1).template cast<Scalar>();

    //         bool flagged = false;
    //         sweep_arrangement.foreach_facet_around_edge(eid, [&](Index fid) {
    //             if (flagged) return;
    //             Index parent = parent_facets(fid);

    //             // Walk the parent's 3 corners (input is triangulated)
    //             const Index pc_begin = envelope.get_facet_corner_begin(parent);
    //             const Index pc_end = envelope.get_facet_corner_end(parent);
    //             const Index pn = pc_end - pc_begin;
    //             for (Index ci = 0; ci < pn; ++ci) {
    //                 if (in_corner_junction[pc_begin + ci] == 0) continue;
    //                 Index pv0 = envelope.get_corner_vertex(pc_begin + ci);
    //                 Index pv1 = envelope.get_corner_vertex(pc_begin + (ci + 1) % pn);
    //                 Point a = in_V.row(pv0);
    //                 Point b = in_V.row(pv1);
    //                 if (on_segment(a, b, q0) && on_segment(a, b, q1)) {
    //                     flagged = true;
    //                     break;
    //                 }
    //             }
    //         });
    //         if (flagged) is_junction[eid] = 1;
    //     }
    // }

    // // Extract feature edges (existing logic)
    // sweep_arrangement.template create_attribute<int8_t>(
    //     "is_feature",
    //     lagrange::AttributeElement::Edge,
    //     lagrange::AttributeUsage::Scalar,
    //     1);
    // auto is_feature = attribute_vector_ref<int8_t>(sweep_arrangement, "is_feature");
    // is_feature.setZero();
    // Index num_edges = sweep_arrangement.get_num_edges();
    // for (Index eid = 0; eid < num_edges; eid++) {
    //     Index edge_valence = sweep_arrangement.count_num_corners_around_edge(eid);
    //     if (edge_valence != 2) {
    //         bool feature_edge = false;
    //         sweep_arrangement.foreach_facet_around_edge(eid, [&](Index fid) {
    //             if (is_valid[fid] != 0) feature_edge = true;
    //         });
    //         if (feature_edge) is_feature[eid] = 1;
    //     }
    // }
    // ============================================================
    // Recover per-edge `edge_label` on the arrangement.
    //   For each output edge eid, find any incident output facet's parent
    //   input facet that has a junction-tagged corner-edge collinear with
    //   this output edge, and copy that corner's int label.
    // ============================================================
    const bool has_corner_label = envelope.has_attribute("corner_edge_label");
    sweep_arrangement.template create_attribute<int32_t>(
        "edge_label",
        lagrange::AttributeElement::Edge,
        lagrange::AttributeUsage::Scalar,
        1);
    auto edge_label = attribute_vector_ref<int32_t>(sweep_arrangement, "edge_label");
    edge_label.setZero();

    if (has_corner_label) {
        auto in_corner_label = attribute_vector_view<int32_t>(envelope, "corner_edge_label");
        auto in_V = vertex_view(envelope).template cast<Scalar>();

        const Scalar tol = Scalar(1e-9);

        auto on_segment = [&](const Point& a, const Point& b, const Point& q) {
            Point ab = b - a, aq = q - a;
            Scalar ab_len2 = ab.squaredNorm();
            if (ab_len2 < std::numeric_limits<Scalar>::epsilon() *
                            std::numeric_limits<Scalar>::epsilon()) {
                return aq.squaredNorm() < tol * tol;
            }
            Scalar cross_norm = ab.cross(aq).norm();
            if (cross_norm > tol * std::sqrt(ab_len2)) return false;
            Scalar t = ab.dot(aq) / ab_len2;
            return t >= -tol && t <= Scalar(1) + tol;
        };

        const Index num_arr_edges = sweep_arrangement.get_num_edges();
        for (Index eid = 0; eid < num_arr_edges; ++eid) {
            auto [v0, v1] = sweep_arrangement.get_edge_vertices(eid);
            Point q0 = out_vertices.row(v0).template cast<Scalar>();
            Point q1 = out_vertices.row(v1).template cast<Scalar>();

            int32_t found_label = 0;
            sweep_arrangement.foreach_facet_around_edge(eid, [&](Index fid) {
                if (found_label != 0) return;
                Index parent = parent_facets(fid);
                const Index pc_begin = envelope.get_facet_corner_begin(parent);
                const Index pc_end = envelope.get_facet_corner_end(parent);
                const Index pn = pc_end - pc_begin;
                for (Index ci = 0; ci < pn; ++ci) {
                    int32_t corner_lbl = in_corner_label[pc_begin + ci];
                    if (corner_lbl == 0) continue;
                    Index pv0 = envelope.get_corner_vertex(pc_begin + ci);
                    Index pv1 = envelope.get_corner_vertex(pc_begin + (ci + 1) % pn);
                    Point a = in_V.row(pv0);
                    Point b = in_V.row(pv1);
                    if (on_segment(a, b, q0) && on_segment(a, b, q1)) {
                        found_label = corner_lbl;
                        break;
                    }
                }
            });
            if (found_label != 0) edge_label[eid] = found_label;
        }
    }

    // ============================================================
    // Existing is_feature computation — also write label 4 onto edges
    // that aren't already labeled by a junction.
    // ============================================================
    sweep_arrangement.template create_attribute<int8_t>(
        "is_feature",
        lagrange::AttributeElement::Edge,
        lagrange::AttributeUsage::Scalar,
        1);
    auto is_feature = attribute_vector_ref<int8_t>(sweep_arrangement, "is_feature");
    is_feature.setZero();
    Index num_edges = sweep_arrangement.get_num_edges();
    for (Index eid = 0; eid < num_edges; eid++) {
        Index edge_valence = sweep_arrangement.count_num_corners_around_edge(eid);
        if (edge_valence != 2) {
            bool feature_edge = false;
            sweep_arrangement.foreach_facet_around_edge(eid, [&](Index fid) {
                if (is_valid[fid] != 0) feature_edge = true;
            });
            if (feature_edge) {
                is_feature[eid] = 1;
                // Tag with label 4 only if no junction label already claimed this edge.
                if (edge_label[eid] == 0) {
                    edge_label[eid] = 4;
                }
            }
        }
    }

    return sweep_arrangement;
}

template <typename Scalar, typename Index>
lagrange::SurfaceMesh<Scalar, Index> extract_sweep_surface_from_arrangement2(
    lagrange::SurfaceMesh<Scalar, Index>& sweep_arrangement)
{
    Index num_arrangement_facets = sweep_arrangement.get_num_facets();
    auto V = vertex_view(sweep_arrangement);
    auto F = facet_view(sweep_arrangement);
    auto is_valid = attribute_vector_view<int8_t>(sweep_arrangement, "valid");
    auto time_values = attribute_vector_view<Scalar>(sweep_arrangement, "time");

    lagrange::SurfaceMesh<Scalar, Index> sweep_surface;
    sweep_surface.add_vertices(
        static_cast<Index>(V.rows()),
        {V.data(), static_cast<size_t>(V.size())});

    Index num_valid_facets = 0;
    for (Index fid = 0; fid < num_arrangement_facets; fid++) {
        if (is_valid[fid] != 0) num_valid_facets++;
    }
    sweep_surface.add_triangles(num_valid_facets);
    auto sweep_F = facet_ref(sweep_surface);

    // Build arrangement_fid -> sweep_fid map for per-facet attribute copy.
    std::vector<Index> sweep_to_arr_fid;
    sweep_to_arr_fid.reserve(num_valid_facets);

    Index count = 0;
    for (Index fid = 0; fid < num_arrangement_facets; fid++) {
        if (is_valid[fid] == 0) {
            continue;
        } else if (is_valid[fid] == 1) {
            sweep_F.row(count) = F.row(fid);
        } else {
            sweep_F.row(count) = F.row(fid).reverse();
        }
        sweep_to_arr_fid.push_back(fid);
        count++;
    }

    // ============================================================
    // Per-corner time (existing)
    // ============================================================
    sweep_surface.template create_attribute<Scalar>(
        "time",
        lagrange::AttributeElement::Corner,
        lagrange::AttributeUsage::Scalar,
        1);
    auto sweep_time_values = attribute_vector_ref<Scalar>(sweep_surface, "time");

    count = 0;
    for (Index fid = 0; fid < num_arrangement_facets; fid++) {
        if (is_valid[fid] == 0) continue;
        const Index sweep_c_begin = sweep_surface.get_facet_corner_begin(count);
        const Index arrang_c_begin = sweep_arrangement.get_facet_corner_begin(fid);
        if (is_valid[fid] == 1) {
            sweep_time_values[sweep_c_begin + 0] = time_values[arrang_c_begin + 0];
            sweep_time_values[sweep_c_begin + 1] = time_values[arrang_c_begin + 1];
            sweep_time_values[sweep_c_begin + 2] = time_values[arrang_c_begin + 2];
        } else {
            sweep_time_values[sweep_c_begin + 0] = time_values[arrang_c_begin + 2];
            sweep_time_values[sweep_c_begin + 1] = time_values[arrang_c_begin + 1];
            sweep_time_values[sweep_c_begin + 2] = time_values[arrang_c_begin + 0];
        }
        count++;
    }

    // ============================================================
    // Per-facet: face_label, face_dom_chunk, envelope_facet_id
    // ============================================================
    if (sweep_arrangement.has_attribute("face_label")) {
        auto in_face_label = attribute_vector_view<uint8_t>(sweep_arrangement, "face_label");
        sweep_surface.template create_attribute<uint8_t>(
            "face_label",
            lagrange::AttributeElement::Facet,
            lagrange::AttributeUsage::Scalar,
            1);
        auto out_face_label = attribute_vector_ref<uint8_t>(sweep_surface, "face_label");
        for (Index sf = 0; sf < num_valid_facets; ++sf) {
            out_face_label[sf] = in_face_label[sweep_to_arr_fid[sf]];
        }
    }
    if (sweep_arrangement.has_attribute("face_dom_chunk")) {
        auto in_face_dom = attribute_vector_view<uint64_t>(sweep_arrangement, "face_dom_chunk");
        sweep_surface.template create_attribute<uint64_t>(
            "face_dom_chunk",
            lagrange::AttributeElement::Facet,
            lagrange::AttributeUsage::Scalar,
            1);
        auto out_face_dom = attribute_vector_ref<uint64_t>(sweep_surface, "face_dom_chunk");
        for (Index sf = 0; sf < num_valid_facets; ++sf) {
            out_face_dom[sf] = in_face_dom[sweep_to_arr_fid[sf]];
        }
    }
    if (sweep_arrangement.has_attribute("envelope_facet_id")) {
        auto in_env_fid = attribute_vector_view<Index>(sweep_arrangement, "envelope_facet_id");
        sweep_surface.template create_attribute<Index>(
            "envelope_facet_id",
            lagrange::AttributeElement::Facet,
            lagrange::AttributeUsage::Scalar,
            1);
        auto out_env_fid = attribute_vector_ref<Index>(sweep_surface, "envelope_facet_id");
        for (Index sf = 0; sf < num_valid_facets; ++sf) {
            out_env_fid[sf] = in_env_fid[sweep_to_arr_fid[sf]];
        }
    }

    // ============================================================
    // Per-edge: edge_label (0 = none, 1/2/3 = junction categories, 4 = feature)
    // ============================================================
    auto pack_edge_key = [](Index a, Index b) -> uint64_t {
        if (a > b) std::swap(a, b);
        return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
    };

    if (sweep_arrangement.has_attribute("edge_label")) {
        auto in_edge_label = attribute_vector_view<int32_t>(sweep_arrangement, "edge_label");

        sweep_surface.initialize_edges();
        sweep_surface.template create_attribute<int32_t>(
            "edge_label",
            lagrange::AttributeElement::Edge,
            lagrange::AttributeUsage::Scalar,
            1);
        auto out_edge_label = attribute_vector_ref<int32_t>(sweep_surface, "edge_label");
        out_edge_label.setZero();

        const Index num_arr_edges = sweep_arrangement.get_num_edges();
        ankerl::unordered_dense::map<uint64_t, int32_t> edge_label_lookup;
        edge_label_lookup.reserve(num_arr_edges / 4);

        for (Index eid = 0; eid < num_arr_edges; ++eid) {
            if (in_edge_label[eid] == 0) continue;
            auto [v0, v1] = sweep_arrangement.get_edge_vertices(eid);
            edge_label_lookup.emplace(pack_edge_key(v0, v1), in_edge_label[eid]);
        }

        const Index num_sweep_edges = sweep_surface.get_num_edges();
        for (Index eid = 0; eid < num_sweep_edges; ++eid) {
            auto [v0, v1] = sweep_surface.get_edge_vertices(eid);
            auto it = edge_label_lookup.find(pack_edge_key(v0, v1));
            if (it != edge_label_lookup.end()) {
                out_edge_label[eid] = it->second;
            }
        }
    }

    // Diagnostic
    size_t per_label[5] = {0, 0, 0, 0, 0};
    if (sweep_surface.has_attribute("edge_label")) {
        auto el = attribute_vector_view<int32_t>(sweep_surface, "edge_label");
        for (Index e = 0; e < sweep_surface.get_num_edges(); ++e) {
            if (el[e] >= 0 && el[e] <= 4) ++per_label[el[e]];
        }
    }
    sweep::logger().info(
        "sweep surface edge labels: 0={}, 1={}, 2={}, 3={}, 4={}",
        per_label[0], per_label[1], per_label[2], per_label[3], per_label[4]);

    // ============================================================
    // Cleanup. NOTE: this renumbers vertices and removes any vertex with
    // no incident facet. Edge attributes survive because lagrange
    // re-indexes them; facet/corner attributes survive too. Per-vertex
    // attributes (none here) would also survive the renumbering.
    // ============================================================
    lagrange::remove_isolated_vertices(sweep_surface);
    return sweep_surface;
}

template <typename Scalar, typename Index>
void save_feature_edges_obj(
    const std::string& filename,
    const lagrange::SurfaceMesh<Scalar, Index>& mesh,
    const std::string& attr_name = "is_feature")
{
    if (!mesh.has_attribute(attr_name)) {
        throw std::runtime_error(
            "Mesh has no '" + attr_name + "' attribute — did the arrangement run?");
    }

    auto flags = lagrange::attribute_vector_view<int8_t>(mesh, attr_name);
    const Index num_edges = mesh.get_num_edges();

    // Collect edges first so we can compact vertices.
    std::vector<std::pair<Index, Index>> kept;
    kept.reserve(num_edges / 8);
    for (Index eid = 0; eid < num_edges; ++eid) {
        if (flags[eid] != 0) {
            auto vts = mesh.get_edge_vertices(eid);
            kept.emplace_back(vts[0], vts[1]);
        }
    }

    // Compact vertex set: only emit vertices that the kept edges actually use.
    ankerl::unordered_dense::map<Index, Index> remap; // mesh idx -> 1-based OBJ idx
    auto V = lagrange::vertex_view(mesh);

    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Cannot open OBJ for writing: " + filename);

    auto register_vertex = [&](Index vi) -> Index {
        auto [it, inserted] = remap.try_emplace(vi, static_cast<Index>(remap.size() + 1));
        if (inserted) {
            out << "v " << V(vi, 0) << ' ' << V(vi, 1) << ' ' << V(vi, 2) << '\n';
        }
        return it->second;
    };

    // First pass writes vertices (interleaved with l-line construction below).
    // To keep the OBJ valid (all v before any l), buffer the line directives.
    std::vector<std::pair<Index, Index>> line_indices;
    line_indices.reserve(kept.size());
    for (auto [v0, v1] : kept) {
        Index a = register_vertex(v0);
        Index b = register_vertex(v1);
        line_indices.emplace_back(a, b);
    }
    for (auto [a, b] : line_indices) {
        out << "l " << a << ' ' << b << '\n';
    }

    out.flush();
    if (!out) throw std::runtime_error("Failed writing OBJ: " + filename);
}

template <typename Scalar, typename Index>
void save_labeled_edges_ply(
    const std::string& filename,
    const lagrange::SurfaceMesh<Scalar, Index>& mesh,
    const std::string& attr_name = "edge_label")
{
    if (!mesh.has_attribute(attr_name)) {
        throw std::runtime_error(
            "Mesh has no '" + attr_name + "' attribute");
    }

    auto labels = lagrange::attribute_vector_view<int32_t>(mesh, attr_name);
    const Index num_edges = mesh.get_num_edges();

    // Collect non-zero-labeled edges with their endpoints.
    struct KeptEdge {
        Index v0, v1;
        int32_t label;
    };
    std::vector<KeptEdge> kept;
    kept.reserve(num_edges / 8);
    for (Index eid = 0; eid < num_edges; ++eid) {
        if (labels[eid] != 0) {
            auto vts = mesh.get_edge_vertices(eid);
            kept.push_back({vts[0], vts[1], labels[eid]});
        }
    }

    // Compact vertices and assign each one a label inherited from the first
    // labeled edge that touched it. If a vertex is shared by two differently-
    // labeled edges, the first-seen label wins (good enough for colormap;
    // junction crossings are rare and hard to color cleanly anyway).
    ankerl::unordered_dense::map<Index, Index> remap;  // mesh idx -> 0-based PLY idx
    std::vector<Index> vert_mesh_idx;
    std::vector<int32_t> vert_label;

    auto register_vertex = [&](Index vi, int32_t edge_lbl) -> Index {
        auto [it, inserted] = remap.try_emplace(vi, static_cast<Index>(vert_mesh_idx.size()));
        if (inserted) {
            vert_mesh_idx.push_back(vi);
            vert_label.push_back(edge_lbl);
        }
        return it->second;
    };

    // Resolve indices first so we can write all `v` entries before any `edge` entries.
    std::vector<std::tuple<Index, Index, int32_t>> edge_indices;
    edge_indices.reserve(kept.size());
    for (const auto& e : kept) {
        Index a = register_vertex(e.v0, e.label);
        Index b = register_vertex(e.v1, e.label);
        edge_indices.emplace_back(a, b, e.label);
    }

    // PLY header
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Cannot open " + filename + " for writing");

    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "comment sweep surface labeled edges (label values: 1/2/3 = junction, 4 = feature)\n";
    out << "element vertex " << vert_mesh_idx.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property float quality\n";
    out << "element edge " << edge_indices.size() << "\n";
    out << "property int vertex1\n";
    out << "property int vertex2\n";
    out << "property int label\n";
    out << "end_header\n";

    // Vertex section
    auto V = lagrange::vertex_view(mesh);
    for (size_t i = 0; i < vert_mesh_idx.size(); ++i) {
        Index vi = vert_mesh_idx[i];
        out << V(vi, 0) << ' '
            << V(vi, 1) << ' '
            << V(vi, 2) << ' '
            << static_cast<float>(vert_label[i]) << '\n';
    }

    // Edge section
    for (const auto& [a, b, lbl] : edge_indices) {
        out << a << ' ' << b << ' ' << lbl << '\n';
    }

    out.flush();
    if (!out) throw std::runtime_error("Failed writing " + filename);
}

template <typename Scalar, typename Index>
void save_sweep_surface_ply(
    const std::string& filename,
    const lagrange::SurfaceMesh<Scalar, Index>& mesh)
{
    if (!mesh.has_attribute("face_dom_chunk")) {
        throw std::runtime_error(
            "Mesh has no 'face_dom_chunk' attribute — was it propagated through the arrangement?");
    }

    auto V = lagrange::vertex_view(mesh);
    auto F = lagrange::facet_view(mesh);
    auto face_dom = lagrange::attribute_vector_view<uint64_t>(mesh, "face_dom_chunk");

    const Index num_vertices = static_cast<Index>(V.rows());
    const Index num_facets = static_cast<Index>(F.rows());

    // Compute per-vertex quality = mean of incident face dom_chunk values.
    // Cast to double for averaging since dom_chunk is a packed uint64.
    std::vector<double> vertex_quality(num_vertices, 0.0);
    std::vector<uint32_t> incident_count(num_vertices, 0);

    for (Index fid = 0; fid < num_facets; ++fid) {
        const double dom_val = static_cast<double>(face_dom[fid]);
        for (Index k = 0; k < 3; ++k) {
            Index vi = static_cast<Index>(F(fid, k));
            vertex_quality[vi] += dom_val;
            incident_count[vi]++;
        }
    }
    for (Index vi = 0; vi < num_vertices; ++vi) {
        if (incident_count[vi] > 0) {
            vertex_quality[vi] /= static_cast<double>(incident_count[vi]);
        }
    }

    // Optional: per-vertex time if the attribute survives in the mesh.
    // const bool has_time = mesh.has_attribute("time") &&
    //     mesh.get_attribute_base("time").get_element_type() == lagrange::AttributeElement::Vertex;
    const bool has_time = mesh.has_attribute("time");
    // Optional per-facet face_label if present.
    const bool has_face_label = mesh.has_attribute("face_label");

    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Cannot open " + filename + " for writing");

    // -------- Header --------
    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "comment sweep surface mesh with per-facet face_dom_chunk and "
           "per-vertex averaged quality\n";
    out << "element vertex " << num_vertices << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property float quality\n";
    if (has_time) {
        out << "property float time\n";
    }
    out << "element face " << num_facets << "\n";
    out << "property list uchar int vertex_indices\n";
    out << "property uint face_dom_chunk\n";
    out << "property float quality\n";
    if (has_face_label) {
        out << "property uchar face_label\n";
    }
    out << "end_header\n";

    // -------- Vertices --------
    if (has_time) {
        auto T = lagrange::attribute_vector_view<Scalar>(mesh, "time");
        for (Index vi = 0; vi < num_vertices; ++vi) {
            out << V(vi, 0) << ' '
                << V(vi, 1) << ' '
                << V(vi, 2) << ' '
                << static_cast<float>(vertex_quality[vi]) << ' '
                << T[vi] << '\n';
        }
    } else {
        for (Index vi = 0; vi < num_vertices; ++vi) {
            out << V(vi, 0) << ' '
                << V(vi, 1) << ' '
                << V(vi, 2) << ' '
                << static_cast<float>(vertex_quality[vi]) << '\n';
        }
    }

    // -------- Faces --------
    if (has_face_label) {
        auto face_label = lagrange::attribute_vector_view<uint8_t>(mesh, "face_label");
        for (Index fid = 0; fid < num_facets; ++fid) {
            out << "3 "
                << F(fid, 0) << ' '
                << F(fid, 1) << ' '
                << F(fid, 2) << ' '
                << static_cast<uint32_t>(face_dom[fid]) << ' '
                << static_cast<float>(face_dom[fid]) << ' '
                << static_cast<unsigned>(face_label[fid]) << '\n';
        }
    } else {
        for (Index fid = 0; fid < num_facets; ++fid) {
            out << "3 "
                << F(fid, 0) << ' '
                << F(fid, 1) << ' '
                << F(fid, 2) << ' '
                << static_cast<uint32_t>(face_dom[fid]) <<' '
                << static_cast<float>(face_dom[fid]) <<  '\n';
        }
    }
    out.flush();
    if (!out) throw std::runtime_error("Failed writing " + filename);
}


#endif /* post_processing_h */
