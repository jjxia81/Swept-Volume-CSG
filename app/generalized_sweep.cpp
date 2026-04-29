// #include <igl/read_triangle_mesh.h>
// #include <igl/signed_distance.h>
#include <lagrange/io/save_mesh.h>
#include <lagrange/mesh_cleanup/remove_degenerate_facets.h>
#include <lagrange/mesh_cleanup/remove_topologically_degenerate_facets.h>
#include <lagrange/topology.h>
#include <lagrange/utils/SmallVector.h>
#include <lagrange/views.h>
#include <sweep/generalized_sweep.h>
#include <sweep/logger.h>
#include <yaml-cpp/yaml.h>
// #include "stf/yaml_parser.h"
#include <stf/stf.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <queue>
#include <span>

#include "trajectory.h"
#include "csgFuncs.h"
#include "csg_tet.h"

#define SAVE_CONTOUR 0
#define batch_stats 0
#define batch_time 0

template <typename Scalar, typename Index>
void save_features(std::string_view filename, lagrange::SurfaceMesh<Scalar, Index>& arrangement)
{
    std::ofstream fout(filename.data());
    if (!fout) {
        throw std::runtime_error("Failed to open file: " + std::string(filename));
    }

    if (!arrangement.has_attribute("is_feature")) {
        throw std::runtime_error("Arrangement mesh does not have 'is_feature' attribute.");
    }
    auto vertex_view = lagrange::vertex_view(arrangement);
    Index num_vertices = arrangement.get_num_vertices();
    for (Index vid = 0; vid < num_vertices; vid++) {
        fout << "v " << vertex_view(vid, 0) << " " << vertex_view(vid, 1) << " "
             << vertex_view(vid, 2) << std::endl;
    }

    auto is_feature = lagrange::attribute_vector_view<int8_t>(arrangement, "is_feature");
    Index num_edges = arrangement.get_num_edges();
    for (Index eid = 0; eid < num_edges; eid++) {
        if (is_feature(eid)) {
            auto [v0, v1] = arrangement.get_edge_vertices(eid);
            fout << "l " << (v0 + 1) << " " << (v1 + 1) << std::endl;
        }
    }
}

// sweep::CSGFunction make_csg_function(const stf::CSGTree<3>& tree)
// {
//     int buf_size = tree.get_eval_buffer_size();
//     auto val_buf = std::make_shared<std::vector<double>>(buf_size);
//     auto idx_buf = std::make_shared<std::vector<int>>(buf_size);

//     return [&tree, val_buf, idx_buf](Eigen::RowVectorXd leaf_values)
//         -> std::pair<double, size_t>
//     {
//         int winner = tree.winning_leaf_flat(
//             leaf_values.data(), val_buf->data(), idx_buf->data());
//         return {leaf_values[winner], static_cast<size_t>(winner)};
//     };
// }

const size_t MAX_CSG_LEAF_NODE_NUM = 256;
sweep::CSGFunction make_csg_function(const stf::CSGTree<3>& tree)
{
    return [&tree](Eigen::RowVectorXd leaf_values)
        -> std::pair<double, size_t>
    {
        // stack-allocated, automatic per-thread, zero overhead
        double val_buf[MAX_CSG_LEAF_NODE_NUM];
        int    idx_buf[MAX_CSG_LEAF_NODE_NUM];
        assert(tree.get_eval_buffer_size() <= MAX_CSG_LEAF_NODE_NUM);
        int winner = tree.winning_leaf_flat(
            leaf_values.data(), val_buf, idx_buf);
        return {leaf_values[winner], static_cast<size_t>(winner)};
    };
}

using FuncType = std::function<std::pair<double, Eigen::RowVector4d>(Eigen::RowVector4d)> ;
std::vector<FuncType> make_leaf_functions(const stf::CSGTree<3>& tree)
{
    auto leaf_funcs = tree.get_leaf_functions();
    
    std::vector<FuncType> funcs;
    funcs.reserve(leaf_funcs.size());

    for (auto* func : leaf_funcs)
    {
        funcs.push_back([func](Eigen::RowVector4d pt) -> std::pair<double, Eigen::RowVector4d>
        {
            std::array<double, 3> pos = {pt[0], pt[1], pt[2]};
            double t = pt[3];

            double val = func->value(pos, t);
            auto grad  = func->gradient(pos, t);  // std::array<Scalar, 4>

            Eigen::RowVector4d grad_eigen;
            grad_eigen << grad[0], grad[1], grad[2], grad[3];

            return {val, grad_eigen};
        });
    }

    return funcs;
}

void load_config(std::string config_file, sweep::GridSpec& grid_spec, sweep::SweepOptions& options)
{
    std::filesystem::path config_path(config_file);
    if (!std::filesystem::exists(config_path) || !std::filesystem::is_regular_file(config_path)) {
        sweep::logger().warn("Configuration file does not exist: {}", config_file);
        return;
    }

    YAML::Node config = YAML::LoadFile(config_path.string());
    if (config["grid"]) {
        auto grid_config = config["grid"];
        if (grid_config["resolution"]) {
            grid_spec.resolution = {
                grid_config["resolution"][0].as<size_t>(),
                grid_config["resolution"][1].as<size_t>(),
                grid_config["resolution"][2].as<size_t>()};
        }
        if (grid_config["bbox_min"]) {
            grid_spec.bbox_min = {
                grid_config["bbox_min"][0].as<float>(),
                grid_config["bbox_min"][1].as<float>(),
                grid_config["bbox_min"][2].as<float>()};
        }
        if (grid_config["bbox_max"]) {
            grid_spec.bbox_max = {
                grid_config["bbox_max"][0].as<float>(),
                grid_config["bbox_max"][1].as<float>(),
                grid_config["bbox_max"][2].as<float>()};
        }
    }
    if (config["parameters"]) {
        auto param_config = config["parameters"];
        if (param_config["epsilon_env"]) {
            options.epsilon_env = param_config["epsilon_env"].as<double>();
        }
        if (param_config["epsilon_sil"]) {
            options.epsilon_sil = param_config["epsilon_sil"].as<double>();
        }
        if (param_config["max_split"]) {
            options.max_split = param_config["max_split"].as<int>();
        }
        if (param_config["with_insideness_check"]) {
            options.with_insideness_check = param_config["with_insideness_check"].as<bool>();
        }
        if (param_config["with_snapping"]) {
            options.with_snapping = param_config["with_snapping"].as<bool>();
        }
        if (param_config["cyclic"]) {
            options.cyclic = param_config["cyclic"].as<bool>();
        }
        if (param_config["volume_threshold"]) {
            options.volume_threshold = param_config["volume_threshold"].as<double>();
        }
        if (param_config["face_count_threshold"]) {
            options.face_count_threshold = param_config["face_count_threshold"].as<size_t>();
        }
        if (param_config["with_adaptive_refinement"]) {
            options.with_adaptive_refinement = param_config["with_adaptive_refinement"].as<bool>();
        }
        if (param_config["initial_time_samples"]) {
            options.initial_time_samples = param_config["initial_time_samples"].as<int>();
        }
        if (param_config["min_tet_radius_ratio"]) {
            options.min_tet_radius_ratio = param_config["min_tet_radius_ratio"].as<double>();
        }
        if (param_config["min_tet_edge_length"]) {
            options.min_tet_edge_length = param_config["min_tet_edge_length"].as<double>();
        }
    }
}

int main(int argc, const char* argv[])
{
    struct
    {
        std::string output_path;
        std::string config_file = "";
        std::string function_file = "";
        double threshold = 0.005;
        double traj_threshold = 0.005;
        int max_splits = std::numeric_limits<int>::max();
        int rot = 0;
        bool without_insideness_check = false;
        bool without_snapping = false;
        bool without_opt_triangulation = false;
        bool cyclic = false;
    } args;
    CLI::App app{"Generalized Swept Volume"};
    app.add_option("output", args.output_path, "Output path")->required();
    app.add_option("-c,--config", args.config_file, "Configuration file");
    app.add_option("-f,--function", args.function_file, "Implicit function file");
    app.add_option("--ee,--epsilon-env", args.threshold, "Envelope threshold");
    app.add_option("--es, --epsilon-sil", args.traj_threshold, "Silhouette threshold");
    app.add_flag(
        "--without-inside-check",
        args.without_insideness_check,
        "Turn on the refinement for the inside regions of the envelope");
    app.add_option("-m,--max-splits", args.max_splits, "Maximum number of splits");
    app.add_option("-r,--rotation-number", args.rot, "Number of rotations");
    app.add_flag(
        "--without-snapping",
        args.without_snapping,
        "Disable vertex snapping in iso-surfacing step");
    app.add_flag(
        "--without-optimal-triangulation",
        args.without_opt_triangulation,
        "Disable optimal triangulation in iso-surfacing triangulation step");
    app.add_flag("--cyclic", args.cyclic, "Whether the trajectory is cyclic or not");
    CLI11_PARSE(app, argc, argv);

    using Scalar = sweep::Scalar;

    std::string output_path = args.output_path;
    int max_splits = args.max_splits;
    bool insideness_check = !args.without_insideness_check;
    std::string function_file = args.function_file;
    double threshold = args.threshold;
    double traj_threshold = args.traj_threshold;
    int rotation = args.rot;
    const int dim = 4;
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    // igl::AABB<Eigen::MatrixXd, 3> tree;
    // igl::FastWindingNumberBVH fwn_bvh;
    
    bool input_csg_funcs = false; 

    std::vector<FuncType> funcs;
    sweep::CSGFunction csg_f; 
    stf::OwnedCSGTree<3> fileCSGFunc;
    stf::ManagedSpaceTimeFunction<3>* managed; 
    std::unique_ptr<stf::SpaceTimeFunction<3>> funPtr;
    stf::CSGTree<3>* csgTreePtr = nullptr;

    if(args.function_file == "")
    {
        args.function_file = "/home/jjxia/Documents/projects/Swept-Volume-CSG/data/csg/tet3d.yaml";
    }
    
    if(args.function_file != "" ) {
        
        if (std::filesystem::exists(args.function_file) &&
            std::filesystem::is_regular_file(args.function_file)) 
        {
            // Parse space-time function with dimension 3
            funPtr = stf::parse_space_time_function_from_file<3>(args.function_file);
            sweep::logger().info("Successfully loaded space-time function from: {}", args.function_file);
            managed = dynamic_cast<stf::ManagedSpaceTimeFunction<3>*>(funPtr.get());
            if (managed) {
                csgTreePtr = dynamic_cast<stf::CSGTree<3>*>(managed->get_function());
            }
            if (!csgTreePtr) {
                sweep::logger().info("The provided space-time function is not a CSG tree.");
                throw std::runtime_error("Expected a CSG tree as input.");
            }
            // fileCSGFunc = stf::YamlParser<3>::parse_csg_from_file(args.function_file);
            // stf::CSGTree<3>* csg = fileCSGFunc.get(); 
            csgTreePtr->build_flat_plan();
            if(!input_csg_funcs)
            {
                csg_f = make_csg_function(*csgTreePtr);
                funcs = make_leaf_functions(*csgTreePtr);
            }
            
           
        }  else {
            sweep::logger().info("The input CSG function file does not exist. ");
        }
    } else {
        /// use hard coded models as default/testing purpose.
        // implicit_sweep = [&](Eigen::RowVector4d data) { return flippingDonutFullTurn(data); };
    }
    
    if (!std::filesystem::exists(output_path)) {
        // Attempt to create the directory
        if (std::filesystem::create_directory(output_path)) {
            sweep::logger().info("Created output directory: {}", output_path);
        } else {
            sweep::logger().error("Failed to create output directory: {}", output_path);
        }
    } else {
        sweep::logger().info("Output directory already exists: {}", output_path);
    }

    sweep::GridSpec grid_spec;
    sweep::SweepOptions options;
    options.out_dir = output_path;

    std::cout << " config file path : " << args.config_file << std::endl;
    if (args.config_file != "") {
        load_config(args.config_file, grid_spec, options);
    } else {

        // Extracting options from command line arguments
        options.epsilon_env = threshold;
        options.epsilon_sil = traj_threshold;
        options.max_split = max_splits;
        options.with_insideness_check = insideness_check;
        options.with_snapping = !args.without_snapping;
        options.cyclic = args.cyclic;
        grid_spec.bbox_max = csgfTet::bbox_max;
        grid_spec.bbox_min = csgfTet::bbox_min;
    }
    
    
    if(input_csg_funcs)
    {
        // funcs.push_back(csgf::sphere3d_f1);
        // funcs.push_back(csgf::sphere3d_f2);
        // grid_spec.bbox_max = csgf::bbox_max;
        // grid_spec.bbox_min = csgf::bbox_min;
        // csg_f = csgf::csgf_sphere3d;

        csg_f = csgfTet::csgf_tet;
        funcs.push_back(csgfTet::tet_f1);
        funcs.push_back(csgfTet::tet_f2);
        funcs.push_back(csgfTet::tet_f3);
        funcs.push_back(csgfTet::tet_f4);
        // grid_spec.bbox_max = csgfTet::bbox_max;
        // grid_spec.bbox_min = csgfTet::bbox_min;
        // funcs.push_back(implicit_sweep);
    } else {
        // funcs.push_back(implicit_sweep);
    }

    if(funcs.empty())
    {
        sweep::logger().info("Warning: no csg leaf node functions.");
        return 0;
    }

    // std::function<std::pair<double, size_t>(Eigen::RowVectorXd)> csg_f = csgf::csgf_sphere3d;  
    // std::function<std::pair<double, size_t>(Eigen::RowVectorXd)> csg_f;

    auto result = sweep::generalized_sweep_csg(funcs, csg_f, csgTreePtr, grid_spec, options);
    // auto& envelope = result.envelope;
    // auto& sweep_surface = result.sweep_surface;
    // auto& sweep_arrangement = result.arrangement;

    // Saving result
    // auto saving_start = std::chrono::time_point_cast<std::chrono::microseconds>(
    //                         std::chrono::high_resolution_clock::now())
    //                         .time_since_epoch()
    //                         .count();
    
    // lagrange::io::save_mesh(output_path + "/sweep_surface.obj", sweep_surface); 
    // lagrange::io::save_mesh(output_path + "/envelope.msh", envelope);
    // lagrange::io::save_mesh(output_path + "/sweep_surface.msh", sweep_surface);
    // lagrange::io::save_mesh(output_path + "/arrangement.msh", sweep_arrangement);
    // save_features(output_path + "/features.obj", sweep_arrangement);

#if SAVE_CONTOUR
    // mtet::save_mesh(output_path + "/tet_grid.msh", grid);
    // save_grid_for_mathematica(output_path + "/contour_iso.json", grid,
    // vertexMap);
#endif

    // auto saving_end = std::chrono::time_point_cast<std::chrono::microseconds>(
    //                       std::chrono::high_resolution_clock::now())
    //                       .time_since_epoch()
    //                       .count();
    // sweep::logger().info("Saving time: {} seconds", (saving_end - saving_start) * 1e-6);

    return 0;
}
