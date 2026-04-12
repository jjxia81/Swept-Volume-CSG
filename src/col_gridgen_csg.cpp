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

bool extractColContour(const std::array<vertexCol*, 4> baseVerts, 
    mtet::MTetMesh& grid, mtet::TetId tid)
{
    const auto& vs = grid.get_tet(tid);
    std::array<mtet::Scalar, 12> spatial_verts; // 3 (xyz) × 4 (verts)
        {
            std::size_t off = 0;
            for (mtet::VertexId corner : vs) { // 'corner' can be a plain id
                std::span<const Scalar, 3> v = grid.get_vertex(corner);
                std::copy_n(v.begin(), 3, spatial_verts.begin() + off);
                off += 3;
            }
        }
        mtetcol::SimplicialColumn<4> column;
        std::array<vertexCol::time_list_f, 4> time = {
            baseVerts[0]->getTimeList_f(),
            baseVerts[1]->getTimeList_f(),
            baseVerts[2]->getTimeList_f(),
            baseVerts[3]->getTimeList_f()};
        std::function<std::span<double>(size_t)> time_func =
            [&](size_t index) -> std::span<double> { return time[index]; };
        std::array<vertexCol::value_list, 4> values = {
            baseVerts[0]->getValueList(),
            baseVerts[1]->getValueList(),
            baseVerts[2]->getValueList(),
            baseVerts[3]->getValueList()};
        std::function<std::span<double>(size_t)> values_func =
            [&](size_t index) -> std::span<double> { return values[index]; };
        column.set_vertices(spatial_verts);
        column.set_simplices(one_column_simp_csg);
        column.set_time_samples(time_func, values_func);
        auto contour = column.extract_contour(0.0, false);
        auto num_polyhedra = contour.get_num_polyhedra();
        auto num_vertices = contour.get_num_vertices();

        std::vector<double> contour_time;
        contour_time.reserve(num_vertices);
        std::vector<int> contour_index;
        contour_index.reserve(num_vertices);
        std::vector<Eigen::RowVector4d> contour_pos;
        contour_pos.reserve(num_vertices);
        parse_vertices(contour, contour_time, contour_index, contour_pos, spatial_verts);
}

// ///see descriptions in header
// bool gridRefineCSG(
//     mtet::MTetMesh& grid,
//     vertExtrude& vertexMap,
//     insidenessMap& insideMap,
//     const std::vector<std::function<std::pair<Scalar, Eigen::RowVector4d>(Eigen::RowVector4d)>> csg_funcs,
//     const double threshold,
//     const double traj_threshold,
//     const int max_splits,
//     const int insideness_check,
//     std::array<double, timer_amount>& profileTimer,
//     std::array<size_t, timer_amount>& profileCount,
//     size_t initial_time_samples,
//     double min_tet_radius_ratio,
//     double min_tet_edge_length)
// {
//     init5CGridCSG(initial_time_samples, grid, csg_funcs, MAX_TIME, vertexMap);
//     double time_scale = calTimeGlobalScaleWithInitGridCSG(vertexMap);

//     std::cout << " --- time scale : " << time_scale << std::endl;

//     double min_tet_ratio = 1.0;
//     ///
//     /// Initiate queue: timeQ and spaceQ
//     auto compTime = [](std::tuple<mtet::Scalar, mtet::TetId, mtet::VertexId, int> timeSub0,
//                        std::tuple<mtet::Scalar, mtet::TetId, mtet::VertexId, int> timeSub1) {
//         return std::get<0>(timeSub0) < std::get<0>(timeSub1);
//     };
//     std::vector<std::tuple<mtet::Scalar, mtet::TetId, mtet::VertexId, int>> timeQ;
//     auto compSpace = [](std::tuple<mtet::Scalar, mtet::TetId, mtet::EdgeId> spaceSub0,
//                         std::tuple<mtet::Scalar, mtet::TetId, mtet::EdgeId> spaceSub1) {
//         return std::get<0>(spaceSub0) < std::get<0>(spaceSub1);
//     };
//     std::vector<std::tuple<mtet::Scalar, mtet::TetId, mtet::EdgeId>> spaceQ;

//     int splits = 0, temporal_splits = 0, spatial_splits = 0;
//     std::array<vertexCol*, 4> baseVerts;
//     std::array<Eigen::RowVector4d, 4> baseCoord;
//     mtet::EdgeId longest_edge;
//     mtet::Scalar longest_edge_length = 0;
//     std::vector<int> timeLenList(MAX_CELL_INTERVALS);
//     std::vector<mtet::Scalar> timeList(MAX_CELL_INTERVALS);
//     std::vector<size_t> indList(MAX_CELL_INTERVALS);
//     std::vector<bool> subList(MAX_CELL_INTERVALS, false);
//     std::vector<bool> choiceList(MAX_CELL_INTERVALS);
//     std::vector<bool> zeroX_list(MAX_CELL_INTERVALS);
//     std::vector<std::vector<size_t>> cellDomFuncIds;
    
//     std::array<vertex4d*, 5> verts{};
//     size_t csg_fn = csg_funcs.size();
//     std::array<vertex4d*, 4> verts_3d{};
    
//     std::array<vertex4d, 4> simple_verts_3d;
//     simple_verts_3d.fill(vertex4d(csg_fn));
//     ///
//     /// Push Queue Function:
//     auto push_one_col = [&](mtet::TetId tid) {
// #if time_profile
//         Timer first_part_timer(first_part, [&](auto timer, auto ms) {
//             combine_timer(profileTimer, profileCount, timer, ms);
//         });
//         Timer first_part_setup_timer(first_part_setup, [&](auto timer, auto ms) {
//             combine_timer(profileTimer, profileCount, timer, ms);
//         });
// #endif
//         const auto& vs = grid.get_tet(tid);
//         //        simpCol colInfo = cell5Map[vs];
//         //        simpCol::cell5_list cell5Col = colInfo.cell5Col;
//         simpCol::cell5_list cell5Col;
//         // Eigen::Matrix<double, 35, Eigen::Dynamic> bezierCol;
//         // sampleCol(vs, vertexMap, cell5Col, baseVerts, verts_list, bezierCol);
//         sampleCol(vs, vertexMap, cell5Col);

//         for (size_t i = 0; i < 4; i++) {
//             baseVerts[i] = &vertexMap[value_of(vs[i])];
//             baseCoord[i] = baseVerts[i]->vert4dList[0].coord;
//         }
//         std::valarray<double> p0(baseCoord[0].data(), 3);
//         std::valarray<double> p1(baseCoord[1].data(), 3);
//         std::valarray<double> p2(baseCoord[2].data(), 3);
//         std::valarray<double> p3(baseCoord[3].data(), 3);
//         auto tet_ratio = tet_radius_ratio({p0, p1, p2, p3});
//         if (tet_ratio < min_tet_radius_ratio) insideMap[vs] = true;
//         /// Compute longest spatial edge
//         longest_edge_length = 0;
//         bool baseSub = false;
//         bool terminate = false;
//         grid.foreach_edge_in_tet(tid, [&](mtet::EdgeId eid, mtet::VertexId v0, mtet::VertexId v1) {
//             auto p0 = grid.get_vertex(v0);
//             auto p1 = grid.get_vertex(v1);
//             mtet::Scalar l = std::sqrt((p0[0] - p1[0]) * (p0[0] - p1[0]) + (p0[1] - p1[1]) * (p0[1] - p1[1]) +
//                              (p0[2] - p1[2]) * (p0[2] - p1[2]) );
//             if (l > longest_edge_length) {
//                 longest_edge_length = l;
//                 longest_edge = eid;
//             }
//         });
// #if time_profile
//         first_part_setup_timer.Stop();
// #endif
//         //        std::vector<int> timeLenList(cell5Col.size());
//         //        std::vector<mtet::Scalar> timeList(cell5Col.size());
//         //        std::vector<size_t> indList(cell5Col.size());
//         //        std::vector<bool> subList(cell5Col.size(), false);
//         //        std::vector<bool> choiceList(cell5Col.size());
//         //        std::vector<bool> zeroX_list(cell5Col.size());
//         //        std::vector<std::array<int, 5>> cell5_index_list(cell5Col.size());
//         bool no_intersect = true;
//         for (size_t cell5It = 0; cell5It < cell5Col.size(); cell5It++) {
//             const auto& simp = cell5Col[cell5It];
//             //            std::array<int, 5> cell5Index = simp.hash;
//             ////            cell5_index_list[cell5It] = cell5Index;
//             //            int lastInd = cell5Index[4];
//             const int* cell5Index = simp.hash.data(); // no copy, same syntax
//             const int lastInd = cell5Index[4];
//             //            const auto& verts = verts_list[cell5It];
//             verts[0] = &baseVerts[lastInd]->vert4dList[cell5Index[lastInd]];
//             size_t ind = 0;
//             for (size_t i = 0; i < 4; i++) {
//                 if (i != lastInd) {
//                     ind++;
//                     verts[ind] = &baseVerts[i]->vert4dList[cell5Index[i]];
//                 }
//             }
//             verts[4] = &baseVerts[lastInd]->vert4dList[cell5Index[lastInd] - 1];
//             // bool inside = false;
//             bool choice = false;
//             bool zeroX = false;
//             //            bool ret = refineFt_new(verts, bezierCol.col(cell5It), traj_threshold,
//             //            inside, choice, zeroX, profileTimer, profileCount);
            
//             bool ret =
//                 refineFtCSG(verts, traj_threshold, choice, zeroX, profileTimer, profileCount,cellDomFuncIds[cell5It]);
//             zeroX_list[cell5It] = zeroX;
//             if(zeroX) no_intersect = false;
//             if (ret) {
//                 subList[cell5It] = true;
//                 timeLenList[cell5It] = (verts[0]->time - verts[4]->time);
//                 timeList[cell5It] = (verts[0]->time + verts[4]->time) / 2;
//                 indList[cell5It] = lastInd;
//                 choiceList[cell5It] = choice;
//                 // choiceList[cell5It] = std::abs(timeLenList[cell5It])  * time_scale / MAX_TIME > longest_edge_length ? true : false;
//             } else {
//                 subList[cell5It] = false;
//             }
//         }
// #if time_profile
//         Timer first_part_setup_timer2(first_part_setup, [&](auto timer, auto ms) {
//             combine_timer(profileTimer, profileCount, timer, ms);
//         });
// #endif
//         for (size_t cell5It = 0; cell5It < cell5Col.size(); cell5It++) {
//             if (subList[cell5It]) {
//                 terminate = true;
//                 if (choiceList[cell5It]) {
//                     if (timeLenList[cell5It] > MIN_TIME) {
//                         timeQ.emplace_back(
//                             timeLenList[cell5It] * time_scale / MAX_TIME,
//                             tid,
//                             vs[indList[cell5It]],
//                             timeList[cell5It]);
//                         std::push_heap(timeQ.begin(), timeQ.end(), compTime);
//                     }
//                 } else {
//                     if (!baseSub) {
//                         if (longest_edge_length > min_tet_edge_length) {
//                             spaceQ.emplace_back(longest_edge_length, tid, longest_edge);
//                             std::push_heap(spaceQ.begin(), spaceQ.end(), compSpace);
//                             baseSub = true;
//                         }
//                     }
//                 }
//             }
//         }
// #if time_profile
//         first_part_setup_timer2.Stop();
//         Timer compute_caps_timer(compute_caps, [&](auto timer, auto ms) {
//             combine_timer(profileTimer, profileCount, timer, ms);
//         });

// #endif

//         if (!baseSub) {
//             for (size_t i = 0; i < 4; i++) {
//                 verts_3d[i] = &baseVerts[i]->vert4dList.front();
//             }
//             bool ret = refine3DCSG(verts_3d, threshold, csg_funcs.size());
//             if (ret) {
//                 if (longest_edge_length > min_tet_edge_length) {
//                     spaceQ.emplace_back(longest_edge_length, tid, longest_edge);
//                     std::push_heap(spaceQ.begin(), spaceQ.end(), compSpace);
//                     baseSub = true;
//                 }
//             }
//         }
//         if (!baseSub) {
//             for (size_t i = 0; i < 4; i++) {
//                 verts_3d[i] = &baseVerts[i]->vert4dList.back();
//             }

//             bool ret = refine3DCSG(verts_3d, threshold, csg_funcs.size());
//             if (ret) {
//                 if (longest_edge_length > min_tet_edge_length) {
//                     spaceQ.emplace_back(longest_edge_length, tid, longest_edge);
//                     std::push_heap(spaceQ.begin(), spaceQ.end(), compSpace);
//                     baseSub = true;
//                 }
//             }
//         }
// #if time_profile
//         compute_caps_timer.Stop();
//         first_part_timer.Stop();
// #endif
//         if (no_intersect) {
//             terminate = true;
//         } else {
//             min_tet_ratio = std::min(min_tet_ratio, tet_ratio);
//         }
//         if (terminate) return;
// #ifndef only_stage1
//     #if time_profile
//         Timer extract_first_iso_timer(extract_first_iso, [&](auto timer, auto ms) {
//             combine_timer(profileTimer, profileCount, timer, ms);
//         });
//     #endif
//         std::array<mtet::Scalar, 12> spatial_verts; // 3 (xyz) × 4 (verts)
//         {
//             std::size_t off = 0;
//             for (mtet::VertexId corner : vs) { // 'corner' can be a plain id
//                 std::span<const Scalar, 3> v = grid.get_vertex(corner);
//                 std::copy_n(v.begin(), 3, spatial_verts.begin() + off);
//                 off += 3;
//             }
//         }
//         mtetcol::SimplicialColumn<4> column;
//         std::array<vertexCol::time_list_f, 4> time = {
//             baseVerts[0]->getTimeList_f(),
//             baseVerts[1]->getTimeList_f(),
//             baseVerts[2]->getTimeList_f(),
//             baseVerts[3]->getTimeList_f()};
//         std::function<std::span<double>(size_t)> time_func =
//             [&](size_t index) -> std::span<double> { return time[index]; };
//         std::array<vertexCol::value_list, 4> values = {
//             baseVerts[0]->getValueList(),
//             baseVerts[1]->getValueList(),
//             baseVerts[2]->getValueList(),
//             baseVerts[3]->getValueList()};
//         std::function<std::span<double>(size_t)> values_func =
//             [&](size_t index) -> std::span<double> { return values[index]; };
//         column.set_vertices(spatial_verts);
//         column.set_simplices(one_column_simp_csg);
//         column.set_time_samples(time_func, values_func);
//         auto contour = column.extract_contour(0.0, false);
//         auto num_polyhedra = contour.get_num_polyhedra();
//         auto num_vertices = contour.get_num_vertices();

//         std::vector<double> contour_time;
//         contour_time.reserve(num_vertices);
//         std::vector<int> contour_index;
//         contour_index.reserve(num_vertices);
//         std::vector<Eigen::RowVector4d> contour_pos;
//         contour_pos.reserve(num_vertices);
//         parse_vertices(contour, contour_time, contour_index, contour_pos, spatial_verts);
//     #if time_profile
//         extract_first_iso_timer.Stop();
//         Timer second_part_timer(second_part, [&](auto timer, auto ms) {
//             combine_timer(profileTimer, profileCount, timer, ms);
//         });
//     #endif
//         for (int i = 0; i < num_polyhedra; i++) {
//             bool simple = contour.is_polyhedron_regular(i);
//             std::vector<mtetcol::Index> vert_id;
//             vert_id.reserve(num_vertices);
//             parse_polyhedron(contour, i, vert_id);
//             if (simple) assert(vert_id.size() == 4);
//             // start traversing the active 5-cell.
//             for (size_t cell5It = 0; cell5It < cell5Col.size(); cell5It++) {
//                 if (!zeroX_list[cell5It]) {
//                     continue;
//                 }
//     #if time_profile
//                 Timer find_intersect_timer(find_intersect, [&](auto timer, auto ms) {
//                     combine_timer(profileTimer, profileCount, timer, ms);
//                 });
//     #endif
//                 int sign = 0;
//                 bool intersect = false;
//                 auto& hash = cell5Col[cell5It].hash;
//                 for (auto& vi : vert_id) {
//                     auto& poly_time = contour_time[vi];
//                     auto& poly_ind = contour_index[vi];
//                     auto tet_time = time[poly_ind][hash[poly_ind]];
//                     compare_time(tet_time, poly_time, intersect, sign);
//                     if (poly_ind == hash[4]) {
//                         tet_time = time[poly_ind][hash[poly_ind] - 1];
//                         compare_time(tet_time, poly_time, intersect, sign);
//                     }
//                 }
//     #if time_profile
//                 find_intersect_timer.Stop();
//     #endif
//                 if (intersect) {
//                     if (simple) {
//     #if time_profile
//                         Timer second_func_timer(second_func, [&](auto timer, auto ms) {
//                             combine_timer(profileTimer, profileCount, timer, ms);
//                         });
//     #endif
//                         std::array<Eigen::RowVector4d, 4> pts;
//                         Eigen::RowVector4d vals;
//                         std::array<Eigen::RowVector4d, 4> grads;
//                         for (int vi = 0; vi < vert_id.size(); vi++) {
//                             auto& vert = simple_verts_3d[vi];
//                             vert.coord = contour_pos[vert_id[vi]];
//                             for(int funcI = 0; funcI < csg_funcs.size(); ++funcI)
//                             {
//                                 auto res = csg_funcs[funcI](vert.coord);
//                                 vert.vals[funcI] = res.first;
//                                 vert.grads.row(funcI) = res.second;
//                             }
//                             verts_3d[vi] = &simple_verts_3d[vi];
//                         }
//                         // if (refine3D(verts_3d, threshold)) {
//                         if (refine3DCSG(verts_3d, threshold, csg_funcs.size())) {
//                             if (longest_edge_length > min_tet_edge_length) {
//                                 spaceQ.emplace_back(longest_edge_length, tid, longest_edge);
//                                 std::push_heap(spaceQ.begin(), spaceQ.end(), compSpace);
//                                 baseSub = true;
//                             }
//                         }
//     #if time_profile
//                         second_func_timer.Stop();
//     #endif
//                     } else if (20 * longest_edge_length > threshold) {
//     #if time_profile
//                         Timer non_simple_poly_timer(non_simple_poly, [&](auto timer, auto ms) {
//                             combine_timer(profileTimer, profileCount, timer, ms);
//                         });
//     #endif
//                         if (longest_edge_length > min_tet_edge_length) {
//                             spaceQ.emplace_back(longest_edge_length, tid, longest_edge);
//                             std::push_heap(spaceQ.begin(), spaceQ.end(), compSpace);
//                             baseSub = true;
//                         }
//     #if time_profile
//                         non_simple_poly_timer.Stop();
//     #endif
//                     }
//                 }
//                 if (terminate) {
//     #if time_profile
//                     second_part_timer.Stop();
//     #endif
//                     return;
//                 };
//             }
//         }
//     #if time_profile
//         second_part_timer.Stop();
//     #endif
// #endif
//     };

//     grid.seq_foreach_tet(
//         [&](mtet::TetId tid, [[maybe_unused]] std::span<const mtet::VertexId, 4> vs) {
//             push_one_col(tid);
//         });
//     std::tuple<mtet::Scalar, mtet::TetId, mtet::VertexId, int> time_ele;
//     std::tuple<mtet::Scalar, mtet::TetId, mtet::EdgeId> space_ele;
//     bool has_poped_time_ele = false; 
//     bool has_poped_space_ele = false;
//     bool refine_temporal = false;
//     while ((!timeQ.empty() || !spaceQ.empty()) && splits < max_splits) {
        
//         if (!timeQ.empty() && !has_poped_time_ele)
//         {
//             std::pop_heap(timeQ.begin(), timeQ.end(), compTime);
//             time_ele = timeQ.back();
//             timeQ.pop_back();
//             has_poped_time_ele = true;
//         }
//         if (!spaceQ.empty() && !has_poped_space_ele)
//         {
//             std::pop_heap(spaceQ.begin(), spaceQ.end(), compSpace);
//             // auto [space_edge_len, tid, eid] 
//             space_ele = spaceQ.back();
//             spaceQ.pop_back();
//             has_poped_space_ele = true;
//         }

//         if(has_poped_time_ele && has_poped_space_ele)
//         {
//             refine_temporal = std::get<0>(time_ele) > std::get<0>(space_ele) ? true : false;
//         } else if(has_poped_time_ele)
//         {
//             refine_temporal = true;
//         } else {
//             refine_temporal = false;
//         }

//         // if(has_poped_time_ele)
//         // {
//         //     refine_temporal = true;
//         // } else {
//         //     refine_temporal = false;
//         // }


//         // if (!timeQ.empty()) 
//         if(refine_temporal)
//         {
//             /// temporal subdivision:
//             // std::pop_heap(timeQ.begin(), timeQ.end(), compTime);
//             // auto [_, tid, vid, time] = timeQ.back();
//             // timeQ.pop_back();
//             has_poped_time_ele = false;
//             auto [_, tid, vid, time] = time_ele;
//             if (!grid.has_tet(tid)) {
//                 continue;
//             }
//             std::span<VertexId, 4> vs = grid.get_tet(tid);
//             vertexCol timeList = vertexMap[value_of(vid)];
//             if (insideMap[vs] || timeList.timeExist[time]) {
//                 continue;
//             }
//             splits++;
//             temporal_splits++;
//             timeList.timeExist[time] = true;
//             vertexCol::vertToSimp newTets;
//             newTets.reserve(timeList.vertTetAssoc.size());
//             for (size_t i = 0; i < timeList.vertTetAssoc.size(); i++) {
//                 auto assocTet = timeList.vertTetAssoc[i];
//                 if (grid.has_tet(assocTet)) {
//                     newTets.emplace_back(assocTet);
//                 }
//             }
//             timeList.vertTetAssoc = newTets;
//             vertex4d newVert(csg_funcs.size());
//             newVert.time = time;
//             newVert.coord = {
//                 timeList.vert4dList[0].coord[0],
//                 timeList.vert4dList[0].coord[1],
//                 timeList.vert4dList[0].coord[2],
//                 (double)time / MAX_TIME};
//             // newVert.valGradList = func(newVert.coord);
//             for(int funcI = 0; funcI < csg_funcs.size(); ++funcI)
//             {
//                 auto res = csg_funcs[funcI](newVert.coord);
//                 newVert.vals[funcI] = res.first;
//                 newVert.grads.row(funcI) = res.second;
//             }
//             timeList.insertTime(newVert);
//             vertexMap[value_of(vid)] = timeList;
//             for (size_t i = 0; i < newTets.size(); i++) {
// #if time_profile
//                 Timer ref_crit_timer(ref_crit, [&](auto timer, auto ms) {
//                     combine_timer(profileTimer, profileCount, timer, ms);
//                 });
// #endif
//                 push_one_col(newTets[i]);
// #if time_profile
//                 ref_crit_timer.Stop();
// #endif
//             }
//         } else {
//             /// spatial subdivision:
//             // std::pop_heap(spaceQ.begin(), spaceQ.end(), compSpace);
//             // auto [space_edge_len, tid, eid] = spaceQ.back();
//             // spaceQ.pop_back();
//             has_poped_space_ele = false;
//             auto [space_edge_len, tid, eid] = space_ele;
//             if (!grid.has_tet(tid)) {
//                 continue;
//             }
//             std::span<VertexId, 4> vs = grid.get_tet(tid);
//             if (insideMap[vs]) {
//                 continue;
//             }
//             splits++;
//             spatial_splits++;
//             vertexCol newVert;
//             std::array<VertexId, 2> vs_old = grid.get_edge_vertices(eid);
//             auto [vid, eid0, eid1] = grid.split_edge(eid);
//             std::span<Scalar, 3> vidCoord = grid.get_vertex(vid);
//             vertexCol v0Col = vertexMap[value_of(vs_old[0])];
//             vertexCol v1Col = vertexMap[value_of(vs_old[1])];
//             vertexCol newVertCol;
//             vertexCol::time_list v0Time = v0Col.getTimeList();
//             vertexCol::time_list v1Time = v1Col.getTimeList();
//             vertexCol::time_list tSamples;
//             std::set_union(
//                 v0Time.begin(),
//                 v0Time.end(),
//                 v1Time.begin(),
//                 v1Time.end(),
//                 std::back_inserter(tSamples));
//             vertexCol::vert4d_list vertColList(tSamples.size()); 
//             for (int i = 0; i < tSamples.size(); i++) {
//                 vertex4d vert(csg_funcs.size());
//                 vert.time = tSamples[i];
//                 double time_fp = (double)vert.time / MAX_TIME;
//                 vert.coord = {vidCoord[0], vidCoord[1], vidCoord[2], time_fp};
//                 // vert.valGradList = func(vert.coord);
//                 for(int funcI = 0; funcI < csg_funcs.size(); ++funcI)
//                 {
//                     auto res = csg_funcs[funcI](vert.coord);
//                     vert.vals[funcI] = res.first;
//                     vert.grads.row(funcI) = res.second;
//                 }
//                 vertColList[i] = vert;
//             }
//             newVertCol.vert4dList = vertColList;
//             vertexMap[value_of(vid)] = newVertCol;
// #if parallel_bezier
//             ///
//             /// Parallel computing for 4D bezier simplex experiments:
//             ///
//             auto push_cols = [&](llvm_vecsmall::SmallVector<mtet::TetId, 256> tetList) {
//                 llvm_vecsmall::SmallVector<std::array<vertex4d*, 5>, 4000> simpList;
//                 for (const auto& tid : tetList) {
//                     const auto& vs = grid.get_tet(tid);
//                     simpCol::cell5_list cell5Col;
//                     sampleCol(vs, vertexMap, cell5Col);
//                     for (size_t cell5It = 0; cell5It < cell5Col.size(); cell5It++) {
//                         std::array<vertex4d*, 5> verts_temp{};
//                         const auto& simp = cell5Col[cell5It];
//                         const int* cell5Index = simp.hash.data();
//                         const int lastInd = cell5Index[4];
//                         verts_temp[0] = &baseVerts[lastInd]->vert4dList[cell5Index[lastInd]];
//                         size_t ind = 0;
//                         for (size_t i = 0; i < 4; i++) {
//                             if (i != lastInd) {
//                                 ind++;
//                                 verts_temp[ind] = &baseVerts[i]->vert4dList[cell5Index[i]];
//                             }
//                         }
//                         verts_temp[4] = &baseVerts[lastInd]->vert4dList[cell5Index[lastInd] - 1];
//                         simpList.emplace_back(verts_temp);
//                     }
//                 }
//                 const int N = (int)simpList.size();
//                 if (N >= 64 * omp_get_max_threads() /*500*/) {
//     #pragma omp parallel
//                     {
//                         const int T = omp_get_num_threads();
//                         const int tid = omp_get_thread_num();
//                         const int blk = (N + T - 1) / T;
//                         const int beg = tid * blk;
//                         const int end = std::min(N, beg + blk);

//                         for (int i = beg; i < end; ++i) {
//                             bool inside = false, choice = false, zeroX = false;
//                             refineFtBezier(simpList[i], traj_threshold, inside, choice, zeroX);
//                         }
//                     }
//                 } else {
//                     // serial fallback
//                     for (int i = 0; i < N; ++i) {
//                         bool inside = false, choice = false, zeroX = false;
//                         refineFtBezier(simpList[i], traj_threshold, inside, choice, zeroX);
//                     }
//                 }
//             };
// #endif

// #if parallel_bezier
//             llvm_vecsmall::SmallVector<mtet::TetId, 256> tetList;
// #endif
//             grid.foreach_tet_around_edge(eid0, [&](mtet::TetId t0) {
//                 std::span<VertexId, 4> vs = grid.get_tet(t0);
//                 for (size_t i = 0; i < 4; i++) {
//                     vertexMap[value_of(vs[i])].vertTetAssoc.emplace_back(t0);
//                 }
//                 insideMap[vs] = false;
// #if time_profile
//                 Timer ref_crit_timer(ref_crit, [&](auto timer, auto ms) {
//                     combine_timer(profileTimer, profileCount, timer, ms);
//                 });
// #endif
// #if parallel_bezier
//                 tetList.emplace_back(t0);
// #endif
//                 push_one_col(t0);
// #if time_profile
//                 ref_crit_timer.Stop();
// #endif
//             });
//             grid.foreach_tet_around_edge(eid1, [&](mtet::TetId t1) {
//                 std::span<VertexId, 4> vs = grid.get_tet(t1);
//                 for (size_t i = 0; i < 4; i++) {
//                     vertexMap[value_of(vs[i])].vertTetAssoc.emplace_back(t1);
//                 }
//                 insideMap[vs] = false;
// #if time_profile
//                 Timer ref_crit_timer(ref_crit, [&](auto timer, auto ms) {
//                     combine_timer(profileTimer, profileCount, timer, ms);
//                 });
// #endif
// #if parallel_bezier
//                 tetList.emplace_back(t1);
// #endif
//                 push_one_col(t1);
// #if time_profile
//                 ref_crit_timer.Stop();
// #endif
//             });
// #if parallel_bezier
//             push_cols(tetList);
// #endif
//         }
//     }
//     sweep::logger().info(
//         "Total splits: {}  Spatial splits: {}  Minimum tet radius ratio: {}",
//         splits,
//         spatial_splits,
//         min_tet_ratio);
//     return true;
// }
