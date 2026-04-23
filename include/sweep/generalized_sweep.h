#pragma once

#include <lagrange/SurfaceMesh.h>

#include <Eigen/Core>
#include <array>
#include <functional>
#include <limits>
#include <filesystem>
#include <string>

namespace sweep {

using Scalar = double;
using Index = uint32_t;

///
/// Type alias for a space-time implicit function.
///
/// The function takes a 4D point (x, y, z, t) as input and returns a pair:
///     - The first element is the scalar value of the implicit function at that
///     point.
///     - The second element is a 4D gradient vector (∂f/∂x, ∂f/∂y, ∂f/∂z,
///     ∂f/∂t) at that point.
///
using SpaceTimeFunction =
    std::function<std::pair<double, Eigen::RowVector4d>(Eigen::RowVector4d)>;

using CSGFunction = std::function<std::pair<double, size_t>(Eigen::RowVectorXd)>;
///
/// Specification for the 3D spatial grid used in the sweep.
///
struct GridSpec {
    /// Resolution of the grid in the x, y, z directions.
    std::array<size_t, 3> resolution = {4, 4, 4};

    // range = {{-.5, 1.5}, {-.75, .75}, {-.75, .75}, {0, 1}};
    /// Axis-aligned bounding box minimum corner.
    // std::array<float, 3> bbox_min = {-0.2f, -0.2f, -0.2f};
    /// Axis-aligned bounding box maximum corner.
    // std::array<float, 3> bbox_max = {1.2f, 1.2f, 1.2f};

    // two sphere range
    // std::array<float, 3> bbox_min = {-0.5f, -0.75f, -0.75f};
    // std::array<float, 3> bbox_max = {1.5f, 0.75f, 0.75f};
    // tet mesh range
    std::array<float, 3> bbox_min = {-1.0f, -1.0f, -1.0f};
    std::array<float, 3> bbox_max = {3.0f, 1.0f, 1.2f};
};

///
/// Result of the generalized sweep operation.
///
struct SweepResult {
    /// The (lifted) envelope mesh.
    ///
    /// The spatial coordinates are stored as vertices of the mesh.
    /// The temporal coordinate is stored as a per-vertex attribute named
    /// "time".
    lagrange::SurfaceMesh<Scalar, Index> envelope;

    /// The spatial arrangement of the envelope.
    ///
    /// The temporal coordinate is stored as a per-corner attribute named
    /// "time".
    ///
    /// The per-facet attribute "valid" indicates the set of facets that are
    /// part of the final sweep surface:
    ///   - 0: the facet is not part of the sweep surface
    ///   - 1: the facet is part of the sweep surface with original orientation
    ///   - -1: the facet is part of the sweep surface with flipped orientation
    ///
    /// The per-edge attribute "is_feature" indicates which edge is part of
    /// feature curves on the sweep surface.
    lagrange::SurfaceMesh<Scalar, Index> arrangement;

    /// The final sweep surface mesh extracted from the arrangement.
    ///
    /// The per-corner attribute "time" stores the temporal coordinate.
    lagrange::SurfaceMesh<Scalar, Index> sweep_surface;
};

///
/// Options for controlling the generalized sweep operation.
///
struct SweepOptions {
    /// Tolerance for envelope grid refinement.
    Scalar epsilon_env = 5e-4;

    /// Tolerance for silhouette grid refinement.
    Scalar epsilon_sil = 5e-3;

    /// Maximum number of splits allowed during grid refinement.
    int max_split = std::numeric_limits<int>::max();

    /// Whether to perform insideness checks during grid refinement.
    /// If true, the algorithm will stop refinement early once it detects a cell
    /// is inside of the swept volume.
    bool with_insideness_check = true;

    /// Whether to enable vertex snapping during isocontouring.
    bool with_snapping = true;

    /// Whether the trajectory is cyclic.
    ///
    /// This feature is experimental and is not fully supported.
    bool cyclic = false;

    /// Minimum volume threshold for arrangement cell filtering.
    ///
    /// Filtered arrangement cells will be merged into adjacent cells during the sweep surface
    /// extraction step.
    Scalar volume_threshold = 1e-5;

    /// Minimum face count threshold for arrangement cell filtering.
    ///
    /// Filtered arrangement cells will be merged into adjacent cells during the sweep surface
    /// extraction step.
    size_t face_count_threshold = 200;

    /// Adaptively refine the input grid based on the implicit function.
    bool with_adaptive_refinement = true;

    /// The input implicit functions are csg functions.
    bool with_csg_funcs = true;

    /// Number of initial uniform time samples per spatial grid vertex.
    size_t initial_time_samples = 8;

    /// The minimum acceptable tetrahedron radius ratio during grid refinement.
    ///
    /// Tets with in-radius to circum-radius ratio below this threshold will not be refined further.
    Scalar min_tet_radius_ratio = 1e-5;

    /// Minimum acceptable tetrahedron edge length during grid refinement.
    ///
    /// Tets with longest edge length below this threshold will not be refined further.
    // Scalar min_tet_edge_length = 2e-5;
    // Scalar min_tet_edge_length = 8e-2;
    Scalar min_tet_edge_length = 0.1;

    std::string out_dir = "." ;
};

///
/// Compute the generalized sweep surface.
///
/// @param f         The space-time implicit function defining the sweep.
/// @param grid_spec Specification of the initial 3D spatial grid.
/// @param options   Options controlling the sweep operation.
///
/// @return          The result of the generalized sweep, including the envelope mesh,
///                  the arrangement mesh, and the final sweep surface.
///
SweepResult generalized_sweep(const std::vector<SpaceTimeFunction>& funcs,
                              CSGFunction csg_f,
                              GridSpec grid_spec = {},
                              SweepOptions options = {});

///
/// Compute the generalized sweep surface from configuration files.
///
/// @param function_file  Path to the file defining the space-time implicit function.
/// @param config_file    Path to the configuration file specifying grid and sweep options.
///
/// @return               The result of the generalized sweep, including the envelope mesh,
///                       the arrangement mesh, and the final sweep surface.
///
SweepResult generalized_sweep_from_config(
        std::filesystem::path function_file,
        std::filesystem::path config_file);

};  // namespace sweep
