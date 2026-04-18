//
//  col_gridgen.cpp
//  adaptive_column_grid
//
//  Created by Yiwen Ju on 12/4/24.
//
// #define only_stage1
#include <functional> // std::reference_wrapper, std::ref

#include <sweep/logger.h>
#include "col_gridgen.h"

#define parallel_bezier 0
std::vector<uint32_t> one_column_simp_csg = {0, 1, 2, 3};



/// @param[in] initial_time_samples: initial number of time samples at each vertex. It will be
/// rounded up to the next power of 2.
/// @param[in] grid: base 3D grid. For each vertex, build a list of time stamps. For each tet, build a list of extruded 4D simplices
/// @param[in] func: the implicit function that represents the swept volume. The input of the function is the 4d coordinate, and the output is an size-4 vector with first entry as the value and the other three as the gradient.
/// @param[in] maxTimeDep: maximum interger-valued time depth of the trajectory. Default: 1024
///
/// @param[out] timeList: a list of time stamps at this vertex
void init5CGridCSG(
    const size_t initial_time_sampels,
    mtet::MTetMesh grid,
    const std::vector<std::function<std::pair<Scalar, Eigen::RowVector4d>(Eigen::RowVector4d)>> csg_funcs,
    const int maxTimeDep,
    vertExtrude& vertexMap)
{
    mtet::Scalar gt_sum = 0;
    mtet::Scalar gs_sum = 0;
    int timeLen = 1;
    // Determine time length as power of 2
    while (timeLen < initial_time_sampels) {
        timeLen <<= 1;
    }
    assert(timeLen > 0);

    int len = maxTimeDep / timeLen;
    vertexCol::time_list time3DList(timeLen + 1);
    for (int i = 0; i < timeLen + 1; i++) {
        int time = i * len;
        time3DList[i] = time;
    }
    size_t csgf_n = csg_funcs.size();
    std::cout << " csgf_n ----------" << csgf_n << std::endl;
    grid.seq_foreach_vertex([&](mtet::VertexId vid, std::span<const mtet::Scalar, 3> data) {
        vertexCol col;
        vertexCol::vert4d_list vertColList(timeLen + 1);
        for (int i = 0; i < timeLen + 1; i++) {
            vertex4d vert(csgf_n);
            vert.time = time3DList[i];
            double time_fp = (double)vert.time / MAX_TIME;
            vert.coord = {data[0], data[1], data[2], time_fp};
            vert.valGradList = csg_funcs[0](vert.coord);
            for(int fi = 0; fi < csgf_n; ++fi)
            {
                auto res = csg_funcs[fi](vert.coord);
                vert.vals[fi] = res.first;
                vert.grads.row(fi) = res.second;
            }
            // std::cout << " vert.grads " << vert.grads << std::endl;
            vertColList[i] = vert;
        }
        col.vert4dList = vertColList;
        vertexMap[value_of(vid)] = col;
    });
    grid.seq_foreach_tet(
        [&](mtet::TetId tid, [[maybe_unused]] std::span<const mtet::VertexId, 4> data) {
            std::span<VertexId, 4> vs = grid.get_tet(tid);
            for (size_t i = 0; i < 4; i++) {
                vertexMap[value_of(vs[i])].vertTetAssoc.push_back(tid);
            }
        });
}


/// @param[in] vertexMap: init grid vertex and function val and gradients.
///
/// @param[out] time_gs: time dimension global scale 
mtet::Scalar calTimeGlobalScaleWithInitGridCSG(vertExtrude& vertexMap)
{
    mtet::Scalar gt_sum = 0;
    mtet::Scalar gs_sum = 0;
    for(auto& ele : vertexMap)
    {
        const auto& cur_col = ele.second;
        const auto& vert_list = cur_col.vert4dList;
        for(const auto& vert : vert_list)
        {
            // std::cout << " vert.grads " << vert.grads.row(0) << std::endl;
            auto grads =  vert.grads.cwiseAbs();
            auto grad_t = grads.col(3).sum() / double(vert.grads.rows());
            gt_sum += grad_t;
            auto grad_x = grads.col(0).sum() / double(vert.grads.rows());
            auto grad_y = grads.col(1).sum() / double(vert.grads.rows());
            auto grad_z = grads.col(2).sum() / double(vert.grads.rows());
       
            gs_sum += (grad_x + grad_y + grad_z) / 3.0;
            // gs_sum += std::sqrt(grad[0] * grad[0] + grad[1]* grad[1] + grad[2] * grad[2]);
        }
    }
    mtet::Scalar time_gs = gt_sum / gs_sum;
    return time_gs;
}

