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

} // namespace cell_complex
