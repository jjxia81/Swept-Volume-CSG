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
    std::vector<double>& verts,
    std::vector<size_t>& simps,
    std::vector<int>& tetActiveTags,
    std::vector<std::vector<double>>& time,
    std::vector<std::vector<double>>& values,
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

#endif /* io_h */
