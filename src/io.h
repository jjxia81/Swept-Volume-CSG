//
//  io.h
//  adaptive_colume_grid
//
//  Created by Yiwen Ju on 12/3/24.
//

#ifndef io_h
#define io_h


#include <mshio/mshio.h>
#include "adaptive_column_grid.h"

#include <fstream>

#include <lagrange/SurfaceMesh.h>
#include <lagrange/views.h>

void convert_4d_grid_col(
    mtet::MTetMesh grid,
    vertExtrude vertexMap,
    std::vector<std::array<double, 3>>& verts,
    std::vector<std::array<size_t, 4>>& simps,
    std::vector<std::vector<double>>& time,
    std::vector<std::vector<double>>& values);


void convert_4d_grid_mtetcol(
    mtet::MTetMesh grid,
    vertExtrude vertexMap,
    std::vector<double>& verts,
    std::vector<size_t>& simps,
    std::vector<std::vector<double>>& time,
    std::vector<std::vector<double>>& values,
    bool cyclic);


void convert_4d_grid_mtetcol(
    mtet::MTetMesh grid,
    vertExtrude vertexMap,
    std::unordered_map<uint64_t, int>& activeColMap,
    std::unordered_map<uint64_t, int>& markTetMap,
    std::unordered_map<uint64_t, std::vector<size_t>>& markTetActive4DtetIdsMap,
    std::vector<double>& verts,
    std::vector<size_t>& simps,
    std::vector<int>& tetActiveTags,
    std::vector<int>& tetMarkTags,
    std::vector<std::vector<size_t>>& tetMarkActive4DtetIds,
    std::vector<std::vector<double>>& time,
    const std::string& out_dir,
    bool cyclic);

mshio::MshSpec generate_spec(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, TimeMap timeMap);

Eigen::VectorXd propagate_labels_bfs(
    const Eigen::MatrixXd& V, // n x 3
    const Eigen::MatrixXi& F, // m x 3
    const TimeMap& timeMap);

void backfill_timeMap_from_labels(
    const Eigen::MatrixXd& V,
    const Eigen::VectorXd& L,
    TimeMap& timeMap);

void save_grid_for_mathematica(
    std::string_view filename,
    mtet::MTetMesh grid,
    vertExtrude vertexMap);

void export_to_mathematica(
    const std::string& filename,
    const std::vector<double>& verts,
    const std::vector<size_t>& simps,
    const std::vector<int>& activeTags,
    const std::vector<std::vector<double>>& time);

void writeGridToJson(
    const std::string& filename,
    const std::vector<double>& verts,
    const std::vector<size_t>& simps,
    const std::vector<int>& activeTags,
    const std::vector<std::vector<double>>& time); 

void write_tets_to_ply(
    const std::vector<double>& verts,
    const std::vector<size_t>& simps,
    const std::string& filepath);

void convert_4d_grid_mtetcol(
    mtet::MTetMesh& grid,
    vertExtrude& vertexMap,
    std::vector<double>& verts4d,       // flat: x,y,z,t per vertex
    std::vector<size_t>& tets4d        // flat: 5 indices per 4D tet (pentatope))
    );

void save_4d_mesh_binary(
    const std::vector<double>& verts4d,
    const std::vector<size_t>& tets4d,
    const std::string& filepath);

void load_4d_mesh_binary(
    std::vector<double>& verts4d,
    std::vector<size_t>& tets4d,
    const std::string& filepath);

void save_column_mesh_binary(
    const std::vector<double>&              verts,
    const std::vector<size_t>&              simps,
    const std::vector<std::vector<double>>& time,
    const std::vector<int>&                 tetMarkTags,
    const std::vector<std::vector<size_t>>& tetActive4DtetIds,
    const std::string&                      filepath);

void load_column_mesh_binary(
    std::vector<double>&              verts,
    std::vector<size_t>&              simps,
    std::vector<std::vector<double>>& time,
    std::vector<size_t>&              marked_tets,
    std::vector<size_t>&              unmarked_tets,
    const std::string&                filepath);

#endif /* io_h */
