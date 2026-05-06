#include <ankerl/unordered_dense.h>
#include <mtet/grid.h>
#include <mtetcol/contour.h>
#include <mtetcol/simplicial_column.h>
#include <nanothread/nanothread.h>
#include <sweep/generalized_sweep.h>
#include <sweep/logger.h>
#include <yaml-cpp/yaml.h>
#include <stf/stf.h>
#include <cell_complex/CellComplex.h>
#include <cell_complex/dominance.h>
#include <cell_complex/generators.h>
#include <cell_complex/algorithm/silhouette.h>
#include <cell_complex/algorithm/envelope.h>

#include <array>
#include <chrono>
#include <vector>
#include <filesystem>

#include "adaptive_column_grid.h"
#include "col_gridgen.h"
#include "io.h"
#include "post_processing.h"
#include "cell_msh_io.h"
#include "cell_obj_io.h"
#include "cell_grid.h"

namespace sweep {

Scalar SweepOptions::time_start = 0.0;
Scalar SweepOptions::time_end = 1.0;

size_t calRefinedGridSampleNumber(vertExtrude& vertexMap)
{
    size_t total_sample_number = 0; 
    for(auto& ele : vertexMap)
    {
        const auto& cur_col = ele.second;
        total_sample_number += cur_col.vert4dList.size();
    }
    return total_sample_number;
}

// std::tuple<
//     std::vector<Scalar>,
//     std::vector<Index>,
//     std::vector<std::vector<Scalar>>,
//     std::vector<std::vector<Scalar>>>
// refine_grid(const SpaceTimeFunction& f, mtet::MTetMesh& grid, const SweepOptions& options)
// {
//     logger().info("Adaptively refine the background grid...");

//     // TODO: investigate why saving and loading is necessary here???
//     mtet::save_mesh("init.msh", grid);
//     grid = mtet::load_mesh("init.msh");
//     std::filesystem::remove("init.msh");

//     vertExtrude vertexMap;
//     insidenessMap insideMap;

//     // TODO: Clarify the purpose of these timers and whether they're still
//     // needed.
//     std::array<double, timer_amount> profileTimer{};
//     std::array<size_t, timer_amount> profileCount{};
//     spdlog::set_level(spdlog::level::off);
//     std::vector<SpaceTimeFunction>  funcs;
//     funcs.push_back(f);
//     if (!gridRefine(
//             grid,
//             vertexMap,
//             insideMap,
//             f,
//             options.epsilon_env,
//             options.epsilon_sil,
//             options.max_split,
//             options.with_insideness_check,
//             profileTimer,
//             profileCount,
//             options.initial_time_samples,
//             options.min_tet_radius_ratio,
//             options.min_tet_edge_length)) {
//         throw std::runtime_error("ERROR: grid generation failed");
//     };

//     spdlog::set_level(spdlog::level::info);
//     size_t total_sample_number = calRefinedGridSampleNumber(vertexMap);
//     sweep::logger().info("--Total sample number {}", total_sample_number);

//     bool cyclic = options.cyclic;
//     std::vector<mtetcol::Scalar> verts;
//     std::vector<mtetcol::Index> simps;
//     std::vector<std::vector<double>> time;
//     std::vector<std::vector<double>> values;
//     // convert_4d_grid_mtetcol(grid, vertexMap, verts, simps, time, values, cyclic);

//     return {verts, simps, time, values};
// }


std::tuple<
    std::vector<Scalar>,
    std::vector<size_t>,
    std::vector<std::vector<Scalar>>,
    std::vector<std::vector<Scalar>>>
refine_grid_csg(const std::vector<SpaceTimeFunction>& csg_funcs, 
    CSGFunction csg_f,
    mtet::MTetMesh& grid, const SweepOptions& options)
{
    logger().info("Adaptively refine the background grid...");

    // TODO: investigate why saving and loading is necessary here???
    mtet::save_mesh("init.msh", grid);
    grid = mtet::load_mesh("init.msh");
    std::filesystem::remove("init.msh");

    vertExtrude vertexMap;
    insidenessMap insideMap;
    std::unordered_map<uint64_t, int> tetActiveMap;

    // TODO: Clarify the purpose of these timers and whether they're still
    // needed.
    std::array<double, timer_amount> profileTimer{};
    std::array<size_t, timer_amount> profileCount{};
    spdlog::set_level(spdlog::level::off);
    // std::vector<SpaceTimeFunction>  funcs;
    // funcs.push_back(f);
    
    if (!gridRefineCSGParallel(
            grid,
            vertexMap,
            insideMap,
            csg_funcs,
            csg_f,
            options.epsilon_env,
            options.epsilon_sil,
            options.max_split,
            options.with_insideness_check,
            profileTimer,
            profileCount,
            tetActiveMap,
            options.initial_time_samples,
            options.min_tet_radius_ratio,
            options.min_tet_edge_length,
            options.out_dir,
            options.time_start,
            options.time_end)) {
        throw std::runtime_error("ERROR: grid generation failed");
    };

    std::string log_path = options.out_dir + "/run_log.txt";
    std::ofstream log_file(log_path);
    auto file_log = [&](const std::string& msg) {
        // std::cout << msg << std::endl;
        if (log_file.is_open()) log_file << msg << std::endl;
    };

    spdlog::set_level(spdlog::level::info);
    size_t total_sample_number = calRefinedGridSampleNumber(vertexMap);
    sweep::logger().info("--Total sample number {}", total_sample_number);
    file_log("--Total sample number " + std::to_string(total_sample_number));
    log_file.close();

    bool cyclic = options.cyclic;
    std::vector<mtetcol::Scalar> verts;
    std::vector<size_t> simps;
    std::vector<int> tetActiveTags;
    std::vector<std::vector<double>> time;
    std::vector<std::vector<double>> values;

    convert_4d_grid_mtetcol(grid, vertexMap, tetActiveMap, verts, simps, tetActiveTags, time, values, options.out_dir, cyclic);

    // std::string outgrid_json_path = options.out_dir +  "/output_grid.json"; 
    // writeGridToJson(outgrid_json_path, verts, simps, tetActiveTags, time);

    // std::string outpath = options.out_dir +  "/output_grid.m"; 
    // export_to_mathematica(outpath, verts, simps, tetActiveTags, time); 
    

    // cell_complex::from_simplicial_columns<4>(verts, simps, timeSamples, timeStartIndices);
    return {verts, simps, time, values};
}

std::tuple<
    std::vector<Scalar>,
    std::vector<Index>,
    std::vector<std::vector<Scalar>>,
    std::vector<std::vector<Scalar>>>
evaluate_grid(const SpaceTimeFunction& f, mtet::MTetMesh& grid, const SweepOptions& options)
{
    size_t num_vertices = grid.get_num_vertices();
    size_t num_tets = grid.get_num_tets();

    std::vector<mtetcol::Scalar> verts(num_vertices * 3);
    std::vector<mtetcol::Index> simps(num_tets * 4);
    std::vector<std::vector<double>> time(num_vertices);
    std::vector<std::vector<double>> values(num_vertices);
    for (size_t i = 0; i < num_vertices; i++) {
        time[i].reserve(options.initial_time_samples);
        values[i].reserve(options.initial_time_samples);
    }

    // Extract vertices
    ankerl::unordered_dense::map<uint64_t, Index> vertex_map;
    vertex_map.reserve(num_vertices);
    grid.seq_foreach_vertex([&](mtet::VertexId vid, std::span<const mtet::Scalar, 3> pos) {
        Index idx = static_cast<Index>(vertex_map.size());
        vertex_map[value_of(vid)] = idx;
        verts[idx * 3 + 0] = static_cast<mtetcol::Scalar>(pos[0]);
        verts[idx * 3 + 1] = static_cast<mtetcol::Scalar>(pos[1]);
        verts[idx * 3 + 2] = static_cast<mtetcol::Scalar>(pos[2]);
    });

    // Extract tets
    size_t tet_count = 0;
    grid.seq_foreach_tet([&](mtet::TetId tid, std::span<const mtet::VertexId, 4> tet_verts) {
        for (size_t i = 0; i < 4; ++i) {
            simps[tet_count * 4 + i] =
                static_cast<mtetcol::Index>(vertex_map[value_of(tet_verts[i])]);
        }
        tet_count++;
    });

    // Extract time and time derivative
    namespace dr = drjit;
    dr::parallel_for(
        dr::blocked_range<size_t>(0, num_vertices, 1),
        [&](dr::blocked_range<size_t> idx_range) {
            for (size_t vid = idx_range.begin(); vid < idx_range.end(); vid++) {
                for (size_t tid = 0; tid < options.initial_time_samples; tid++) {
                    double t = static_cast<double>(tid) / (options.initial_time_samples - 1);
                    Eigen::RowVector4d eval_point;
                    eval_point << verts[vid * 3 + 0], verts[vid * 3 + 1], verts[vid * 3 + 2], t;
                    auto eval = f(eval_point);
                    time[vid].push_back(t);
                    values[vid].push_back(eval.second[3]); // Value is time derivative.
                }
            }
        });

    return {verts, simps, time, values};
}

lagrange::SurfaceMesh<Scalar, Index> compute_envelope(
    const SpaceTimeFunction& f,
    mtetcol::Contour<4>& contour,
    const SweepOptions& options)
{
    constexpr int dim = 4;
    size_t num_contour_vertices = contour.get_num_vertices();
    std::vector<double> function_values(num_contour_vertices);
    std::vector<double> gradient_values(num_contour_vertices * dim);
    for (size_t i = 0; i < num_contour_vertices; ++i) {
        auto pos = contour.get_vertex(i);
        auto pos_eval = f(Eigen::RowVector4d{pos[0], pos[1], pos[2], pos[3]});
        function_values[i] = pos_eval.first;
        gradient_values[dim * i] = pos_eval.second[0];
        gradient_values[dim * i + 1] = pos_eval.second[1];
        gradient_values[dim * i + 2] = pos_eval.second[2];
        gradient_values[dim * i + 3] = pos_eval.second[3];
    }

    // Extract isocontour
    auto isocontour = contour.isocontour(function_values, gradient_values, options.with_snapping);
    if (!isocontour.is_manifold()) {
        throw std::runtime_error("ERROR: extracted isocontour is not manifold");
    }
    if (isocontour.get_num_cycles() == 0) {
        throw std::runtime_error("ERROR: extracted isocontour has zero cycles");
    }
    isocontour.triangulate_cycles(true);
    lagrange::SurfaceMesh<Scalar, Index> envelope = isocontour_to_mesh<Scalar, Index>(isocontour);
    envelope.initialize_edges();

    return envelope;
}

void log_config(const GridSpec& grid_spec, const SweepOptions& options)
{
    sweep::logger().info("=== Generalized Sweep Parameters ===");
    sweep::logger().info(
        "Grid resolution: {} x {} x {}",
        grid_spec.resolution[0],
        grid_spec.resolution[1],
        grid_spec.resolution[2]);
    sweep::logger().info(
        "Grid bbox min: ({}, {}, {})",
        grid_spec.bbox_min[0],
        grid_spec.bbox_min[1],
        grid_spec.bbox_min[2]);
    sweep::logger().info(
        "Grid bbox max: ({}, {}, {})",
        grid_spec.bbox_max[0],
        grid_spec.bbox_max[1],
        grid_spec.bbox_max[2]);
    sweep::logger().info("Envelope epsilon: {}", options.epsilon_env);
    sweep::logger().info("Silhouette epsilon: {}", options.epsilon_sil);
    sweep::logger().info("Max splits: {}", options.max_split);
    sweep::logger().info("Insideness check: {}", options.with_insideness_check);
    sweep::logger().info("Vertex snapping: {}", options.with_snapping);
    sweep::logger().info("Cyclic trajectory: {}", options.cyclic);
    sweep::logger().info("Volume threshold: {}", options.volume_threshold);
    sweep::logger().info("Face count threshold: {}", options.face_count_threshold);
    sweep::logger().info("Adaptive refinement: {}", options.with_adaptive_refinement);
    sweep::logger().info("Initial time samples: {}", options.initial_time_samples);
    sweep::logger().info("Minimum tet radius ratio: {}", options.min_tet_radius_ratio);
    sweep::logger().info("Minimum tet edge length: {}", options.min_tet_edge_length);
    sweep::logger().info("=====================================");
}

void load_config(std::filesystem::path config_path,
        sweep::GridSpec& grid_spec, sweep::SweepOptions& options)
{
    if (!std::filesystem::exists(config_path) || !std::filesystem::is_regular_file(config_path)) {
        sweep::logger().warn("Configuration file does not exist: {}", config_path.string());
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

using FaceKey = cell_complex::Cell<2, 4>::KeyType;
using EdgeKey = cell_complex::Cell<1, 4>::KeyType;

void extraceFeatureLinesFromCC(cell_complex::CellComplex<4> &ccSelect, 
                            const size_t root_start, const size_t num_leafs,
                            std::vector<EdgeKey>& junction_edges,
                            std::vector<int>& edges_labels)
{
    // Build edge-raw-index → neighbouring face keys
    
    cell_complex::compute_dominance(ccSelect);

    ankerl::unordered_dense::map<size_t, std::vector<FaceKey>> edge_to_faces;
    for (const auto& [face_key, face_ref] : ccSelect.get_cells<2>().items()) {
        for (const auto& bd_key : face_ref.get().boundary) {
            edge_to_faces[cell_complex::get_index<1, 4>(bd_key)].push_back(face_key);
        }
    }
    // Helper:generate_edge_label by their face labels 
    // face labels : 1, 2, 3, 4; 
    // {*, 3} or {3, *} -> 0
    // cap - cap {1, 1} or {2, 2} -> 1
    // cap - EqF {1, 4}, {4, 1}, {2, 4} or {4, 2} -> 2
    // EqF - EqF {2, 2} -> 3
    auto generate_edge_label = [](const cell_complex::Cell<2, 4>& f0, const cell_complex::Cell<2, 4>& f1) {
        if(f0.label == 3 || f1.label == 3) return 0;
        if(f0.label == 2 && f1.label == 2) return 1;
        if(f0.label == 1 && f1.label == 1) return 1;
        if(f0.label + f1.label == 3 ) return 1;
        if(f0.label + f1.label == 5 || f0.label + f1.label == 6) return 2;  
        if(f0.label == 4 && f1.label == 4) return 3;
        return 0; 
    };
    auto edge_map = ccSelect.get_cells<1>().items();
    auto is_silhouette = [](const cell_complex::Cell<2, 4>& f)
    {
        return f.label == 3;
    };
    
    for (const auto& [edge_key, edge_ref] : ccSelect.get_cells<1>().items()) {
        size_t raw = cell_complex::get_index<1, 4>(edge_key);
        auto it = edge_to_faces.find(raw);
        if (it == edge_to_faces.end() || it->second.size() != 2) continue;
   
        const auto& f0 = ccSelect.get_cell<2>(it->second[0]);
        const auto& f1 = ccSelect.get_cell<2>(it->second[1]);
        if( is_silhouette(f0) || is_silhouette(f1))
        continue;
        auto f0DomData = cell_complex::get_chunk(f0.dominance, root_start, num_leafs);
        auto f1DomData = cell_complex::get_chunk(f1.dominance, root_start, num_leafs);
        bool diff_sweep =  f0DomData != f1DomData;
        if (diff_sweep) 
        {
            junction_edges.push_back(edge_key);
            int edge_label = generate_edge_label(f0, f1);
            edges_labels.push_back(edge_label);
        }
    }
    logger().info("Found {} junction edges ({} sil-sil + sil-cap)",
                junction_edges.size(), junction_edges.size());
}

SweepResult generalized_sweep_csg(const std::vector<SpaceTimeFunction>& funcs, 
        CSGFunction csg_f, 
        stf::CSGTree<3>* csgTreePtr,
        GridSpec grid_spec, 
        SweepOptions options)
{
    log_config(grid_spec, options);

    auto init_grid_start = std::chrono::high_resolution_clock::now();
    SweepResult result;
    auto grid =
        mtet::generate_tet_grid(grid_spec.resolution, grid_spec.bbox_min, grid_spec.bbox_max);
    auto init_grid_end = std::chrono::high_resolution_clock::now();
    logger().info(
        "Initial grid generation time: {} seconds",
        std::chrono::duration<double>(init_grid_end - init_grid_start).count());

    std::vector<mtetcol::Scalar> verts;
    std::vector<size_t> simps;
    std::vector<std::vector<double>> time;
    std::vector<std::vector<double>> values;

    if (options.with_adaptive_refinement) {
        if(options.with_csg_funcs)
        {
            auto refine_start = std::chrono::high_resolution_clock::now();
            std::tie(verts, simps, time, values) = refine_grid_csg(funcs, csg_f, grid, options);
            auto refine_end = std::chrono::high_resolution_clock::now();
            logger().info(
                "Grid refinement time: {} seconds",
                std::chrono::duration<double>(refine_end - refine_start).count());
        } 
    } else {
        // auto evaluate_start = std::chrono::high_resolution_clock::now();
        // std::tie(verts, simps, time, values) = evaluate_grid(funcs[0], grid, options);
        // auto evaluate_end = std::chrono::high_resolution_clock::now();
        // logger().info(
        //     "Grid evaluation time: {} seconds",
        //     std::chrono::duration<double>(evaluate_end - evaluate_start).count());
    }
    values.clear(); // not used 

    // std::function<std::span<double>(size_t)> time_func = [&](size_t index) -> std::span<double> {
    //     return time[index];
    // };
    // std::function<std::span<double>(size_t)> values_func = [&](size_t index) -> std::span<double> {
    //     return values[index];
    // };

    auto sf_start = std::chrono::high_resolution_clock::now();

    std::vector<double> timeSamples; 
    std::vector<size_t> timeStartIndices;
    timeStartIndices.push_back(0);
    size_t timeEndIndex = 0;
   
    for(size_t i = 0; i < time.size(); ++i)
    {
        for(auto tVal : time[i])
        {
            timeSamples.push_back(tVal);
        }
        timeEndIndex += time[i].size();
        timeStartIndices.push_back(timeEndIndex);
    }

    auto cellFromGrid = cell_complex::from_simplicial_columns<4>(verts, simps, timeSamples, timeStartIndices);
    logger().info("Successfully generated column grid");
    verts.clear(); simps.clear();timeSamples.clear();
    timeStartIndices.clear(); time.clear();
    auto saving_start = std::chrono::time_point_cast<std::chrono::microseconds>(
                            std::chrono::high_resolution_clock::now())
                            .time_since_epoch()
                            .count();

    auto &ccSelect = cellFromGrid;
    // bool use_config_file = false;
    // if(use_config_file)
    // {
    //     std::string grid_config_path = "/home/jjxia/Documents/projects/Swept-Volume-CSG/data/csg/config_test.yaml";
    //     auto ccFromFile = cell_complex::load_uniform_grid<4>(grid_config_path);
    //     logger().info("cell complex successfully generated from config file");
    //     ccSelect =   ccFromFile;
    // }
    

    cell_complex::algorithm::compute_silhouette_complex(ccSelect, *csgTreePtr);
    // cell_complex::save_obj(options.out_dir + "/silhouette.obj", ccSelect);
    // cell_complex::save_msh(options.out_dir + "/silhouette.msh", ccSelect);
    cellFromGrid.validate();

    // Alternative: select edges where the root-level dominance spans >1 leaf
    size_t num_leafs = funcs.size();
    size_t root_start = csgTreePtr->get_root_index() * num_leafs;

    cell_complex::algorithm::compute_envelope_complex(ccSelect, *csgTreePtr);
    
    auto sf_end = std::chrono::high_resolution_clock::now();
    logger().info(
                "cell complex surfacing time: {} seconds",
                std::chrono::duration<double>(sf_end - sf_start).count());
    
    
    auto feature_start = std::chrono::high_resolution_clock::now();
    // Convert envelope cell complex -> lagrange mesh with "time" attribute
    // Triangulate polygonal 2-cells in place
    cell_complex::triangulate_all_2cells(ccSelect);

    // cell_complex::save_obj(options.out_dir + "/envelope.obj", ccSelect);
    // cell_complex::save_msh(options.out_dir + "/envelope.msh", ccSelect);

    // auto envelope_mesh = envelope_complex_to_mesh<Scalar, uint32_t>(ccSelect);
    // auto envelope_mesh = envelope_complex_to_mesh<Scalar, uint32_t>
    //         (ccSelect, root_start, num_leafs, junction_raw_ids);
    // cell_complex::compute_dominance(ccSelect);
    std::vector<EdgeKey> junction_edges;
    std::vector<int> edges_labels;
    extraceFeatureLinesFromCC(ccSelect, root_start, num_leafs, junction_edges, edges_labels);
    auto feature_end = std::chrono::high_resolution_clock::now();
    logger().info(
                "feature lines generation from cc envelop time: {} seconds",
                std::chrono::duration<double>(feature_end - feature_start).count());
    cell_complex::save_edges_to_ply<4>(
    options.out_dir + "/feature_edges_from_ccEnvelop.ply",
    junction_edges,
    edges_labels,
    ccSelect);
    // cell_complex::save_edges_to_obj(options.out_dir + "/feature_edges_from_ccEnvelop.obj", junction_edges, ccSelect);
    cell_complex::save_edges_to_mathematica(options.out_dir + "/feature_edges_from_ccEnvelop.m",
    junction_edges,
    edges_labels,
    ccSelect);
    auto saving_end = std::chrono::time_point_cast<std::chrono::microseconds>(
                          std::chrono::high_resolution_clock::now())
                          .time_since_epoch()
                          .count();
    logger().info("Surfacing and save time: {} seconds", (saving_end - saving_start) * 1e-6);

    // ankerl::unordered_dense::set<size_t> junction_raw_ids;
    // junction_raw_ids.reserve(junction_edges.size());
    // for (const auto& ek : junction_edges) {
    //     junction_raw_ids.insert(cell_complex::get_index<1, 4>(ek));
    // }
    auto ma_start = std::chrono::high_resolution_clock::now();
    ankerl::unordered_dense::map<size_t, int> junction_label_map;
    junction_label_map.reserve(junction_edges.size());
    for (size_t i = 0; i < junction_edges.size(); ++i) {
        junction_label_map.emplace(
            cell_complex::get_index<1, 4>(junction_edges[i]),
            edges_labels[i]);
    }
    
    auto envelope_mesh = envelope_complex_to_mesh2<Scalar, uint32_t>
            (ccSelect, root_start, num_leafs, junction_label_map);

    save_sweep_surface_ply<Scalar, uint32_t>(
    options.out_dir + "/envelop_surface_labeled.ply",
    envelope_mesh);
    // debug_dump_mesh_with_time_ply<Scalar, uint32_t>(
    // options.out_dir + "/debug_01_envelope_mesh.ply", envelope_mesh);

    // Compute arrangement (self-intersection resolution + cell labeling)
    result.arrangement = compute_envelope_arrangement2<Scalar, uint32_t>(
        envelope_mesh,
        options.volume_threshold,
        options.face_count_threshold);
    
    // envelope_mesh.clear_facets();
    // envelope_mesh.clear_edges();
    // envelope_mesh.clear_vertices();

    // debug_dump_mesh_with_time_ply<Scalar, uint32_t>(
    // options.out_dir + "/debug_02_arrangement.ply", result.arrangement);

    auto ma_end = std::chrono::high_resolution_clock::now();
    logger().info(
                "Mesh arrangement time: {} seconds",
                std::chrono::duration<double>(ma_end - ma_start).count());
    // Extract the valid (winding-number boundary) facets as the sweep surface
    result.sweep_surface = extract_sweep_surface_from_arrangement2<Scalar, uint32_t>(
        result.arrangement);

    // debug_dump_mesh_with_time_ply<Scalar, uint32_t>(
    // options.out_dir + "/debug_03_sweep_surface.ply", result.sweep_surface);

    auto sweep_surf_end = std::chrono::high_resolution_clock::now();
    logger().info(
                "Sweep surface extraction time: {} seconds",
                std::chrono::duration<double>(sweep_surf_end - ma_end).count());
    // logger().info("Extract surface time: {} seconds", (sweep_surf_end - ma_end) * 1e-6);

    // After result.sweep_surface is finalized:
    // save_feature_edges_obj(options.out_dir + "/sweep_features_lines.obj", result.sweep_surface);

    // And/or for is_junction:
    // save_feature_edges_obj(
    //     options.out_dir + "/sweep_junctions_lines.obj",
    //     result.sweep_surface,
    //     "is_junction");
    
    save_labeled_edges_ply<Scalar, uint32_t>(
    options.out_dir + "/sweep_feature_edges_labeled.ply",
    result.sweep_surface);

    save_labeled_edges_msh<Scalar, uint32_t>(
    options.out_dir + "/sweep_feature_edges_labeled.msh",
    result.sweep_surface);

    save_sweep_surface_ply<Scalar, uint32_t>(
    options.out_dir + "/sweep_surface_labeled.ply",
    result.sweep_surface);

    save_sweep_surface_msh<Scalar, uint32_t>(options.out_dir + "/sweep_surface_labeled.msh", result.sweep_surface);
        
    
    return result;
}

SweepResult generalized_sweep_from_config(
        std::filesystem::path function_file,
        std::filesystem::path config_file,
        std::string out_dir)
{
    if (!std::filesystem::exists(function_file)) {
        throw std::runtime_error("ERROR: sweep file does not exist: " + function_file.string());
    }
    if (!std::filesystem::exists(config_file)) {
        throw std::runtime_error("ERROR: options file does not exist: " + config_file.string());
    }

    sweep::GridSpec grid_spec;
    sweep::SweepOptions options;
    load_config(config_file, grid_spec, options);

    auto funPtr = stf::parse_space_time_function_from_file<3>(function_file.string());

    stf::CSGTree<3>* csgTreePtr = nullptr;
    auto* managed = dynamic_cast<stf::ManagedSpaceTimeFunction<3>*>(funPtr.get());
    if (managed) {
        csgTreePtr = dynamic_cast<stf::CSGTree<3>*>(managed->get_function());
    }
    if (!csgTreePtr) {
        throw std::runtime_error("Expected a CSG tree as input: " + function_file.string());
    }

    csgTreePtr->build_flat_plan();

    auto leaf_funcs = csgTreePtr->get_leaf_functions();
    std::vector<SpaceTimeFunction> funcs;
    funcs.reserve(leaf_funcs.size());
    for (auto* func : leaf_funcs) {
        funcs.push_back([func](Eigen::RowVector4d pt) -> std::pair<double, Eigen::RowVector4d> {
            std::array<double, 3> pos = {pt[0], pt[1], pt[2]};
            double t = pt[3];
            double val = func->value(pos, t);
            auto grad = func->gradient(pos, t);
            Eigen::RowVector4d grad_eigen;
            grad_eigen << grad[0], grad[1], grad[2], grad[3];
            return {val, grad_eigen};
        });
    }
    
    CSGFunction csg_f = [csgTreePtr](Eigen::RowVectorXd leaf_values) -> std::pair<double, size_t> {
        int buf_size = csgTreePtr->get_eval_buffer_size();
        std::vector<double> val_buf(buf_size);
        std::vector<int> idx_buf(buf_size);
        int winner = csgTreePtr->winning_leaf_flat(
            leaf_values.data(), val_buf.data(), idx_buf.data());
        return {leaf_values[winner], static_cast<size_t>(winner)};
    };

    options.out_dir = out_dir;
    return generalized_sweep_csg(funcs, csg_f, csgTreePtr, grid_spec, options);
}

} // namespace sweep
