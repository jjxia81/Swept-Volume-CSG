//
//  col_gridgen_csg.cpp
//  adaptive_column_grid
//
//  Created by Jianjun Xia on 10/4/26.
//
#include <functional> // std::reference_wrapper, std::ref
#include <sweep/logger.h>
#include "col_gridgen.h"
#include <unordered_set>
#include "bezier_simplex.h"
#include <omp.h>
#define parallel_bezier 0

// ============================================================
//  Types & comparators (file-scope so they're shared by all helpers)
// ============================================================

using TimeElem  = std::tuple<mtet::Scalar, mtet::TetId, mtet::VertexId, int>;
using SpaceElem = std::tuple<mtet::Scalar, mtet::TetId, mtet::EdgeId>;
using CSGFuncs = std::vector<std::function<std::pair<Scalar, Eigen::RowVector4d>(Eigen::RowVector4d)>>;  
std::vector<uint32_t> one_column_simp_opt = {0, 1, 2, 3};

// shared data 
// Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor> bezierValsShared;
// Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor> bezierFtValsShared;
std::unordered_map<uint64_t, int>* colActiveMapPtr = nullptr;

// status data for temporal refinement
bool isTemporalRefine = false;
uint64_t tempRefineVtId = 0; 
int  tempRefineTimeVal = 0;
double time_start_G = 0;
double time_end_G = 1.0;

bool refine_using_openmp = false; 

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
    // std::cout << " csgf_n ----------" << csgf_n << std::endl;
    grid.seq_foreach_vertex([&](mtet::VertexId vid, std::span<const mtet::Scalar, 3> data) {
        vertexCol col;
        vertexCol::vert4d_list vertColList(timeLen + 1);
        for (int i = 0; i < timeLen + 1; i++) {
            vertex4d vert(csgf_n);
            vert.time = time3DList[i];
            double time_fp = (double)vert.time / MAX_TIME;
            // double time_fp = (double)vert.time / MAX_TIME * (time_end_G - time_start_G) + time_start_G;
            vert.coord = {data[0], data[1], data[2], time_fp};
            for(int fi = 0; fi < csgf_n; ++fi)
            {
                auto res = csg_funcs[fi](vert.coord);
                vert.vals[fi] = res.first;
                vert.grads.row(fi) = res.second;
            }
            vert.valGradList = csg_funcs[0](vert.coord);
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

static auto compTime = [](const TimeElem& a, const TimeElem& b) {
    return std::get<0>(a) < std::get<0>(b);
};
static auto compSpace = [](const SpaceElem& a, const SpaceElem& b) {
    return std::get<0>(a) < std::get<0>(b);
};

struct PairHash {
    size_t operator()(const std::pair<size_t, size_t>& p) const {
        return std::hash<size_t>{}(p.first) ^ (std::hash<size_t>{}(p.second) << 32);
    }
};




// std::unordered_map<mtet::TetId, bool>* colActiveMapPtr
// ============================================================
//  Scratch state shared across all push_one_col calls.
//  Kept in one struct so it is easy to pass around and avoids
//  repeated heap allocations inside the hot loop.
// ============================================================

struct ColScratch {
    std::array<vertexCol*, 4>          baseVertsPtr{};
    std::array<Eigen::RowVector4d, 4>  baseCoord{};
    mtet::EdgeId                        longest_edge{};
    mtet::Scalar                        longest_edge_length = 0;
    std::array<vertex4d*, 5>            tet4DVertsPtr{};
    std::array<vertex4d,  4>            polygonVerts;
    std::array<vertex4d*, 4>            polygonVertsPtr{};
    std::array<vertex4d*, 4>            capVertsPtr{};
    std::vector<int>         timeLenList = std::vector<int>(MAX_CELL_INTERVALS);
    std::vector<mtet::Scalar> timeList   = std::vector<mtet::Scalar>(MAX_CELL_INTERVALS);
    std::vector<size_t>      indList     = std::vector<size_t>(MAX_CELL_INTERVALS);
    std::vector<bool>        subList     = std::vector<bool>(MAX_CELL_INTERVALS, false);
    std::vector<bool>        choiceList  = std::vector<bool>(MAX_CELL_INTERVALS);
    std::vector<bool>        zeroX_list  = std::vector<bool>(MAX_CELL_INTERVALS);
    std::vector<std::unordered_set<size_t>> cellDomFuncIds;
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> cellDFuncFt0XIds;
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> cellDFunc0XIds;
};

// ============================================================
//  Small pure helpers
// ============================================================

/// Populate baseVertsPtr[] and baseCoord[] for the four corners of a tet.
static void gather_base_verts(
    const std::span<VertexId, 4>&      vs,
    vertExtrude&                        vertexMap,
    std::array<vertexCol*, 4>&          baseVertsPtr,
    std::array<Eigen::RowVector4d, 4>& baseCoord)
{
    for (size_t i = 0; i < 4; i++) {
        if(vertexMap.find(value_of(vs[i])) == vertexMap.end())
        {
            std::cout << " vs id is not in the map : " << value_of(vs[i]) << std::endl;
        }
        baseVertsPtr[i] = &vertexMap[value_of(vs[i])];
        baseCoord[i] = baseVertsPtr[i]->vert4dList[0].coord;
    }
}

static void set_base_verts_active(
    const std::span<VertexId, 4>&      vs,
    vertExtrude&                        vertexMap)
{
    for (size_t i = 0; i < 4; i++) {
        vertexMap[value_of(vs[i])].isActive = true;
    }
}

/// Walk all edges of a tet and return the longest one.
static void find_longest_edge(
    mtet::MTetMesh& grid,
    mtet::TetId     tid,
    mtet::EdgeId&   longest_edge,
    mtet::Scalar&   longest_edge_length)
{
    longest_edge_length = 0;
    grid.foreach_edge_in_tet(tid, [&](mtet::EdgeId eid, mtet::VertexId v0, mtet::VertexId v1) {
        auto p0 = grid.get_vertex(v0);
        auto p1 = grid.get_vertex(v1);
        mtet::Scalar l = std::sqrt(
            (p0[0]-p1[0])*(p0[0]-p1[0]) +
            (p0[1]-p1[1])*(p0[1]-p1[1]) +
            (p0[2]-p1[2])*(p0[2]-p1[2]));
        if (l > longest_edge_length) {
            longest_edge_length = l;
            longest_edge        = eid;
        }
    });
}

/// Fill verts[0..4] from a cell5 entry and its base vertex columns.
static void bind_cell5_verts(
    const cell5&                       simp,
    const std::array<vertexCol*, 4>&   baseVertsPtr,
    std::array<vertex4d*, 5>&          vertsPtr)
{
    const int* idx    = simp.hash.data();
    const int  lastInd = idx[4];
    vertsPtr[0] = &baseVertsPtr[lastInd]->vert4dList[idx[lastInd]];
    size_t ind = 0;
    for (size_t i = 0; i < 4; i++)
        if (i != (size_t)lastInd)
            vertsPtr[++ind] = &baseVertsPtr[i]->vert4dList[idx[i]];
    vertsPtr[4] = &baseVertsPtr[lastInd]->vert4dList[idx[lastInd] - 1];
}

/// Push a spatial refinement entry (no-op if already pushed or edge too short).
static bool try_push_space(
    std::vector<SpaceElem>& spaceQ,
    bool&                   baseSub,
    mtet::TetId             tid,
    mtet::EdgeId            longest_edge,
    mtet::Scalar            longest_edge_length,
    double                  min_tet_edge_length)
{
    if (baseSub || longest_edge_length <= min_tet_edge_length) return false;
    spaceQ.emplace_back(longest_edge_length, tid, longest_edge);
    std::push_heap(spaceQ.begin(), spaceQ.end(), compSpace);
    baseSub = true;
    return true;
}

// ============================================================
//  Stage-2 (contour) helpers
// ============================================================

static void collect_spatial_verts(
    mtet::MTetMesh&               grid,
    const std::span<VertexId, 4>& vs,
    std::array<mtet::Scalar, 12>& out)
{
    std::size_t off = 0;
    for (mtet::VertexId corner : vs) {
        std::span<const Scalar, 3> v = grid.get_vertex(corner);
        std::copy_n(v.begin(), 3, out.begin() + off);
        off += 3;
    }
}

static mtetcol::Contour<4> extract_column_contour(
    const std::array<vertexCol*, 4>&         baseVertsPtr,
    std::array<mtet::Scalar, 12>&      spatial_verts,
    std::vector<uint32_t>&   one_column_simp,
    const size_t domfId)
{
    std::array<vertexCol::time_list_f, 4> time = {
        baseVertsPtr[0]->getTimeList_f(), baseVertsPtr[1]->getTimeList_f(),
        baseVertsPtr[2]->getTimeList_f(), baseVertsPtr[3]->getTimeList_f()};
    std::function<std::span<double>(size_t)> time_func =
        [&](size_t i) -> std::span<double> { return time[i]; };

    std::array<vertexCol::value_list, 4> values = {
        baseVertsPtr[0]->getFtValueList(domfId), baseVertsPtr[1]->getFtValueList(domfId),
        baseVertsPtr[2]->getFtValueList(domfId), baseVertsPtr[3]->getFtValueList(domfId)};
    std::function<std::span<double>(size_t)> values_func =
        [&](size_t i) -> std::span<double> { return values[i]; };

    mtetcol::SimplicialColumn<4> column;
    column.set_vertices(spatial_verts);
    column.set_simplices(one_column_simp_opt);
    column.set_time_samples(time_func, values_func);
    return column.extract_contour(0.0, false);
}


static mtetcol::Contour<4> extract_column_contour(
    const std::array<vertexCol*, 4>&         baseVertsPtr,
    std::array<mtet::Scalar, 12>&      spatial_verts,
    std::vector<uint32_t>&   one_column_simp,
    const size_t domfId1, const size_t domfId2)
{
    std::array<vertexCol::time_list_f, 4> time = {
        baseVertsPtr[0]->getTimeList_f(), baseVertsPtr[1]->getTimeList_f(),
        baseVertsPtr[2]->getTimeList_f(), baseVertsPtr[3]->getTimeList_f()};
    std::function<std::span<double>(size_t)> time_func =
        [&](size_t i) -> std::span<double> { return time[i]; };

    std::array<vertexCol::value_list, 4> values = {
        baseVertsPtr[0]->getEFValueList(domfId1, domfId2), 
        baseVertsPtr[1]->getEFValueList(domfId1, domfId2),
        baseVertsPtr[2]->getEFValueList(domfId1, domfId2), 
        baseVertsPtr[3]->getEFValueList(domfId1, domfId2)};

    std::function<std::span<double>(size_t)> values_func =
        [&](size_t i) -> std::span<double> { return values[i]; };

    mtetcol::SimplicialColumn<4> column;
    column.set_vertices(spatial_verts);
    column.set_simplices(one_column_simp_opt);
    column.set_time_samples(time_func, values_func);
    return column.extract_contour(0.0, false);
}

static bool cell5_intersects_poly(
    const cell5&                                   simp,
    const std::vector<mtetcol::Index>&             vert_id,
    const std::vector<double>&                     contour_time,
    const std::vector<int>&                        contour_index,
    const std::array<vertexCol::time_list_f, 4>&   time)
{
    int  sign = 0;
    bool intersect = false;
    const auto& hash = simp.hash;
    for (auto vi : vert_id) {
        const double poly_time = contour_time[vi];
        const int    poly_ind  = contour_index[vi];
        compare_time(time[poly_ind][hash[poly_ind]], poly_time, intersect, sign);
        if (intersect) return true;
        if (poly_ind == hash[4]) {
            compare_time(time[poly_ind][hash[poly_ind] - 1], poly_time, intersect, sign);
            if (intersect) return true;
        }
    }
    return intersect;
}

// ============================================================
//  Spatial split: build the new midpoint vertex column
// ============================================================

static vertexCol make_new_spatial_vert(
    mtet::MTetMesh&  grid,
    mtet::VertexId   vid,
    const vertexCol& v0Col,
    const vertexCol& v1Col,
    const CSGFuncs&  funcs)
{
    std::span<Scalar, 3>  vidCoord = grid.get_vertex(vid);
    vertexCol::time_list  v0Time   = v0Col.getTimeList();
    vertexCol::time_list  v1Time   = v1Col.getTimeList();
    // vertexCol::time_list  v0Time   = v0Col.getActiveTimeList();
    // vertexCol::time_list  v1Time   = v1Col.getActiveTimeList();
    vertexCol::time_list  tSamples;
    std::set_union(v0Time.begin(), v0Time.end(),
                   v1Time.begin(), v1Time.end(),
                   std::back_inserter(tSamples));

    vertexCol::vert4d_list vertColList(tSamples.size());
    for (size_t i = 0; i < tSamples.size(); i++) {
        vertex4d vert(funcs.size());
        vert.time        = tSamples[i];
        const double tfp = (double)vert.time / MAX_TIME;
        // double tfp = (double)vert.time / MAX_TIME * (time_end_G - time_start_G) + time_start_G;
        vert.coord       = {vidCoord[0], vidCoord[1], vidCoord[2], tfp};
        for(size_t dfi = 0; dfi < funcs.size(); ++dfi )
        {
            auto cur_val_grad = funcs[dfi](vert.coord);
            vert.vals[dfi] = cur_val_grad.first;
            vert.grads.row(dfi) = cur_val_grad.second;
        }
        vert.valGradList = {vert.vals[0], vert.grads.row(0)}; 
        vertColList[i]   = vert;
    }
    vertexCol col;
    col.vert4dList = std::move(vertColList);
    return col;
}

// ============================================================
//  push_one_col  — one self-contained function instead of a lambda
//  (captures nothing; everything passed explicitly)
// ============================================================

/// Context bundle to avoid a very long parameter list.
struct PushOneColCtx {
    mtet::MTetMesh&     grid;
    vertExtrude&        vertexMap;
    insidenessMap&      insideMap;
    const CSGFuncs&     funcs;
    const double        threshold;
    const double        traj_threshold;
    const int           insideness_check;
    const double        time_scale;
    const double        min_tet_radius_ratio;
    const double        min_tet_edge_length;
    std::vector<TimeElem>&  timeQ;
    std::vector<SpaceElem>& spaceQ;
    std::array<double, timer_amount>& profileTimer;
    std::array<size_t, timer_amount>& profileCount;
    double&             min_tet_ratio;
    ColScratch&         scratch;
};



uint64_t getTetKeyByVids(const std::span<VertexId, 4>& vs)
{
    std::array<uint64_t, 4> ids = {value_of(vs[0]), value_of(vs[1]), value_of(vs[2]), value_of(vs[3])};
    std::sort(ids.begin(), ids.end());
    // combine into single hash
    uint64_t key = ids[0];
    key ^= ids[1] + 0x9e3779b9 + (key << 6) + (key >> 2);
    key ^= ids[2] + 0x9e3779b9 + (key << 6) + (key >> 2);
    key ^= ids[3] + 0x9e3779b9 + (key << 6) + (key >> 2);
    return key;
}

static void push_one_col(mtet::TetId tid, PushOneColCtx& ctx)
{
    auto& [grid, vertexMap, insideMap, funcs,
           threshold, traj_threshold, insideness_check,
           time_scale, min_tet_radius_ratio, min_tet_edge_length,
           timeQ, spaceQ, profileTimer, profileCount,
           min_tet_ratio, sc] = ctx;

#if time_profile
    Timer first_part_timer(first_part, [&](auto t, auto ms) {
        combine_timer(profileTimer, profileCount, t, ms); });
    Timer first_part_setup_timer(first_part_setup, [&](auto t, auto ms) {
        combine_timer(profileTimer, profileCount, t, ms); });
#endif

    const size_t CSGFuncNum = funcs.size();
    if(!grid.has_tet(tid)) return;
    const auto& tetVids = grid.get_tet(tid);

    // std::cout << "start sampleCol " << std::endl;
    simpCol::cell5_list cell5Col;
    sampleCol(tetVids, vertexMap, cell5Col);

    // std::cout << "start get  gather_base_verts " << std::endl;
    gather_base_verts(tetVids, vertexMap, sc.baseVertsPtr, sc.baseCoord);

    // std::cout << "finish get  gather_base_verts " << std::endl;

    // Tet quality — mark degenerate tets as inside and skip further work
    {
        // std::valarray<double> p0(sc.baseCoord[0].data(), 3);
        // std::valarray<double> p1(sc.baseCoord[1].data(), 3);
        // std::valarray<double> p2(sc.baseCoord[2].data(), 3);
        // std::valarray<double> p3(sc.baseCoord[3].data(), 3);
        // std::cout << " p0  " << p0[0] << " " << p0[1] << " " << p0[2]<< std::endl;
        // std::cout << " p1  " << p1[0] << " " << p1[1] << " " << p1[2]<< std::endl;
        // std::cout << " p2  " << p2[0] << " " << p2[1] << " " << p2[2]<< std::endl;
        // std::cout << " p3  " << p3[0] << " " << p3[1] << " " << p3[2]<< std::endl;

        std::array<Eigen::RowVector3d, 4> tet_pts = {
                sc.baseCoord[0].head<3>(),
                sc.baseCoord[1].head<3>(),
                sc.baseCoord[2].head<3>(),
                sc.baseCoord[3].head<3>()
            };
        const auto tet_ratio = tet_radius_ratio(tet_pts);
        // std::array<std::valarray<double>, 4> tet_pts{p0, p1, p2, p3};
        // const auto tet_ratio = tet_radius_ratio(tet_pts);

        // std::cout << " tet_ratio  " << tet_ratio << std::endl; 
        if (tet_ratio < min_tet_radius_ratio) {
            insideMap[tetVids] = true;
            return;
        }
        min_tet_ratio = std::min(min_tet_ratio, tet_ratio);
    }

    // std::cout << "to find find_longest_edge " << std::endl;
    find_longest_edge(grid, tid, sc.longest_edge, sc.longest_edge_length);
    // std::cout << "finish find_longest_edge " << std::endl;

#if time_profile
    first_part_setup_timer.Stop();
#endif

    // ------------------------------------------------------------------
    //  Stage 1.1: evaluate each cell5 for silhouette temporal / spatial refinement
    // ------------------------------------------------------------------
    bool terminate    = false;
    bool baseSub      = false;
    bool activeCol    = false;
    // sc.cellDomFuncIds.resize(cell5Col.size());
    sc.cellDomFuncIds.assign(cell5Col.size(), {});
    sc.cellDFuncFt0XIds.setZero(cell5Col.size(), CSGFuncNum);
    sc.cellDFunc0XIds.setZero(cell5Col.size(), CSGFuncNum);
    
    // std::vector<Eigen::MatrixXi> equalSurfaceFuncPairsCol(CSGFuncNum); 
    std::vector<Eigen::MatrixXi> equalSurfaceFuncPairs(CSGFuncNum, Eigen::MatrixXi::Zero(CSGFuncNum, CSGFuncNum));
    std::unordered_map<std::pair<size_t, size_t>, std::vector<size_t>, PairHash> pairToTets;
    size_t last_v_ind = 0;
    if(isTemporalRefine)
    {
        for(size_t tvi = 0; tvi < 4; ++ tvi )
        {
            if(value_of(tetVids[tvi]) == tempRefineVtId)
            {
                last_v_ind = tvi;
            }
        }
    }  
    
    
    std::vector<bool> cell5TempUpdated(cell5Col.size(), false);
    Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor> bezierValsShared;
    Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor> bezierFtValsShared;
    bezierValsShared.setZero(CSGFuncNum, 35);
    bezierFtValsShared.setZero(CSGFuncNum, 35);
    
    for (size_t ci = 0; ci < cell5Col.size(); ci++) {
        if(isTemporalRefine)
        {
            bool isUpdatedTempTet = false;
            if (cell5Col[ci].bot(last_v_ind) == tempRefineTimeVal ||
            (cell5Col[ci].top() == tempRefineTimeVal && cell5Col[ci].hash[4] == last_v_ind)) {
                isUpdatedTempTet = true;
                cell5TempUpdated[ci] = true;
            }
            if(!isUpdatedTempTet) continue;
        }

        // std::cout << "start to bind cells " << std::endl;
        bind_cell5_verts(cell5Col[ci], sc.baseVertsPtr, sc.tet4DVertsPtr);
        
        std::vector<size_t> domFIds;
        calBezierCoordsAndDomFuncIds(sc.tet4DVertsPtr, profileTimer, profileCount, bezierValsShared, bezierFtValsShared, domFIds);
        
        bool choice = false, zeroX = false;
        bool inside = false;
        bool needs_refine = false;
        double max_ef_error = 0;
        needs_refine  =
            refineFtCSG(sc.tet4DVertsPtr, bezierValsShared, bezierFtValsShared, domFIds, traj_threshold, choice, inside, zeroX,
                     profileTimer, profileCount, sc.cellDFuncFt0XIds.row(ci), sc.cellDFunc0XIds.row(ci));
   
        sc.zeroX_list[ci] = zeroX;
        if(insideness_check) {
            if (inside) {
                insideMap[tetVids] = true;
                return;
            }
        }
        
        // std::cout << " successful refineFtCSG " << std::endl;
        if(zeroX) activeCol = true;
        bool eqaulSurf0X = false;
        bool refineB3 = true;
        std::unordered_set<size_t> domEqualFuncIds;
        if(!needs_refine && refineB3)
        {
            for (size_t id_a = 0; id_a < CSGFuncNum; ++id_a)
            {
                if(sc.cellDFunc0XIds.row(ci)(id_a) == 0) continue;
                for (size_t id_b = id_a + 1; id_b < CSGFuncNum; ++id_b)
                {
                    if(sc.cellDFunc0XIds.row(ci)(id_b) == 0) continue;
                    needs_refine = refineEqualSurfaceCSG(sc.tet4DVertsPtr, bezierValsShared,bezierFtValsShared,
                    traj_threshold, {id_a, id_b}, choice, eqaulSurf0X,max_ef_error, profileTimer,profileCount);
                    if(eqaulSurf0X) 
                    {
                        activeCol = true;
                        // equalSurfaceFuncPairs[ci](id_a, id_b) = 1;
                        pairToTets[{id_a, id_b}].push_back(ci);
                        domEqualFuncIds.insert(id_a);
                        domEqualFuncIds.insert(id_b);
                    }
                    if(needs_refine) break;
                }
                if(needs_refine) break;
            }
        }

        // if(!needs_refine && domEqualFuncIds.size() >= 3)
        // {
        //     std::vector<size_t> ids(domEqualFuncIds.begin(), domEqualFuncIds.end());
        //     const size_t n_f = ids.size();
        //     for (size_t i = 0; i < n_f; ++i) {
        //         for (size_t j = i + 1; j < n_f; ++j) {
        //             for (size_t k = j + 1; k < n_f; ++k) {
        //                 std::array<size_t, 3> tripleFuncIds{ids[i], ids[j], ids[k]};
        //                 needs_refine = refineTripleSurfaceCSG(
        //                     sc.tet4DVertsPtr, bezierValsShared,bezierFtValsShared,
        //                     traj_threshold,tripleFuncIds,profileTimer, profileCount);
        //                 if(needs_refine) break;
        //             }
        //             if(needs_refine) break;
        //         }
        //         if(needs_refine) break;
        //     }
        // }

        if (needs_refine) {
            sc.subList[ci]     = true;
            sc.timeLenList[ci] = sc.tet4DVertsPtr[0]->time - sc.tet4DVertsPtr[4]->time;
            sc.timeList[ci]    = (sc.tet4DVertsPtr[0]->time + sc.tet4DVertsPtr[4]->time) / 2;
            sc.indList[ci]     = cell5Col[ci].hash[4];
            sc.choiceList[ci]  = choice;
            sc.choiceList[ci]  = sc.timeLenList[ci] /double(MAX_TIME) * time_scale 
                                > sc.longest_edge_length;
            
        } else {
            sc.subList[ci] = false;
            

        }
    }
  
#if time_profile
    Timer first_part_setup_timer2(first_part_setup, [&](auto t, auto ms) {
        combine_timer(profileTimer, profileCount, t, ms); });
#endif
    // Push refinement requests
    for (size_t ci = 0; ci < cell5Col.size(); ci++) {
        if (!sc.subList[ci]) continue;
        terminate = true;
        if (sc.choiceList[ci]) 
        {
            if (sc.timeLenList[ci] > MIN_TIME) {
                
                // double dt_sum = 0; double dx_sum = 0; double dy_sum = 0; double dz_sum = 0; 
                // for(size_t fi = 0; fi < sc.cellDFunc0XIds.cols(); ++ fi)
                // {
                //     if(sc.cellDFunc0XIds(ci,fi) != 0)
                //     {
                //         bind_cell5_verts(cell5Col[ci], sc.baseVertsPtr, sc.tet4DVertsPtr);
                //         const auto& g1s = sc.tet4DVertsPtr[0]->grads;
                //         const auto& g2s = sc.tet4DVertsPtr[1]->grads;
                //         const auto& g3s = sc.tet4DVertsPtr[2]->grads;
                //         const auto& g4s = sc.tet4DVertsPtr[3]->grads;
                //         const auto& g5s = sc.tet4DVertsPtr[4]->grads;
                //         dt_sum += abs(g1s(fi,3)) + abs(g2s(fi,3)) + abs(g3s(fi,3)) + abs(g4s(fi,3)) + abs(g5s(fi,3));
                //         dx_sum += abs(g1s(fi,0)) + abs(g2s(fi,0)) + abs(g3s(fi,0)) + abs(g4s(fi,0)) + abs(g5s(fi,0));
                //         dy_sum += abs(g1s(fi,1)) + abs(g2s(fi,1)) + abs(g3s(fi,1)) + abs(g4s(fi,1)) + abs(g5s(fi,1));
                //         dz_sum += abs(g1s(fi,2)) + abs(g2s(fi,2)) + abs(g3s(fi,2)) + abs(g4s(fi,2)) + abs(g5s(fi,2));
                //     }
                // }
                // double local_time_scale = std::max(1.0, (dt_sum + 0.000001)/( (dx_sum + dy_sum + dz_sum)/ 3 + 0.000001));
                
                timeQ.emplace_back(
                    (double)sc.timeLenList[ci] * time_scale / MAX_TIME,
                    tid, tetVids[sc.indList[ci]], (int)sc.timeList[ci]);
                std::push_heap(timeQ.begin(), timeQ.end(), compTime);
                // break;
            }
        } else {
            if(!baseSub)
            {
                if(sc.longest_edge_length > min_tet_edge_length)
                {
                    try_push_space(spaceQ, baseSub, tid,
                           sc.longest_edge, sc.longest_edge_length, min_tet_edge_length);
                } else if(sc.timeLenList[ci] > MIN_TIME)
                {
                    timeQ.emplace_back(
                    (double)sc.timeLenList[ci] * time_scale / MAX_TIME,
                    tid, tetVids[sc.indList[ci]], (int)sc.timeList[ci]);
                    std::push_heap(timeQ.begin(), timeQ.end(), compTime);
                }
                
            }
            // break;
        }
    }

    // std::cout << " successful push refine result " << std::endl;
    
    // if(activeCol) set_base_verts_active(tetVids, vertexMap);
    // if((*colActiveMapPtr).find(value_of(tid)))
    // (*colActiveMapPtr)[value_of(tid)] = activeCol? 1: 0;

#if time_profile
    first_part_setup_timer2.Stop();
    Timer compute_caps_timer(compute_caps, [&](auto t, auto ms) {
        combine_timer(profileTimer, profileCount, t, ms); });
#endif
    // terminate = true;
    bool refineCap = true;
    if (!baseSub && !terminate && refineCap 
        && (!isTemporalRefine || (isTemporalRefine && cell5TempUpdated[0])))
    {    
        for(size_t dfId = 0; dfId < funcs.size(); ++dfId)
        {    
            if(sc.cellDFunc0XIds.row(0)(dfId) == 0) continue;
            activeCol = true;
            for (size_t i = 0; i < 4; i++)
            {
                sc.capVertsPtr[i] = &sc.baseVertsPtr[i]->vert4dList.front(); 
            }
            if (refine3DCSG(sc.capVertsPtr, threshold, dfId))
            {
                try_push_space(spaceQ, baseSub, tid,
                    sc.longest_edge, sc.longest_edge_length, min_tet_edge_length);
                break;
            }
        }
    }
    
    if (!baseSub && !terminate && refineCap 
        && (!isTemporalRefine || (isTemporalRefine && cell5TempUpdated[cell5Col.size()-1]))) 
    {
        for(size_t dfId = 0; dfId < funcs.size(); ++dfId)
        {
            if(sc.cellDFunc0XIds.row(cell5Col.size()-1)(dfId) == 0) continue; 
            activeCol = true;
            for (size_t i = 0; i < 4; i++)
            {
                sc.capVertsPtr[i] = &sc.baseVertsPtr[i]->vert4dList.back();
            }
            if (refine3DCSG(sc.capVertsPtr, threshold, dfId))
            {
                try_push_space(spaceQ, baseSub, tid,
                    sc.longest_edge, sc.longest_edge_length, min_tet_edge_length);
                break;
            }
        }
    }
    if(activeCol)
    {
        auto tetKey = getTetKeyByVids(tetVids);
        (*colActiveMapPtr)[tetKey] = 1;
        for (size_t vid = 0; vid < 5; vid++)
        {
            sc.tet4DVertsPtr[vid]->active = true;
        }
    }

#if time_profile
    compute_caps_timer.Stop();
    first_part_timer.Stop();
#endif

    if(baseSub) terminate = true;

    // std::cout << " before terminiate " <<  std::endl;
    // if(terminate) std::cout << " is to  terminiate " <<  std::endl;
    // terminate = true;
    // if(!activeCol) terminate = true;
    // terminate = true;
    if (terminate) return;

    // ------------------------------------------------------------------
    //  Stage 2: contour-based refinement
    // ------------------------------------------------------------------
#ifndef only_stage1
#if time_profile
    Timer extract_first_iso_timer(extract_first_iso, [&](auto t, auto ms) {
        combine_timer(profileTimer, profileCount, t, ms); });
#endif

    std::array<mtet::Scalar, 12> spatial_verts;
    collect_spatial_verts(grid, tetVids, spatial_verts);

    // std::cout << " collect_spatial_verts " <<  std::endl;

    // std::unordered_set<size_t> unique_domfIds;
    // for (const auto& vec : sc.cellDomFuncIds) {
    //     unique_domfIds.insert(vec.begin(), vec.end());
    // }

    bool refineB2 = true;
    std::vector<int> nonZeroColIds;
    for (int j = 0; j < sc.cellDFuncFt0XIds.cols(); ++j)
        if (sc.cellDFuncFt0XIds.col(j).any())
            nonZeroColIds.push_back(j);
    // std::cout << " nonZeroColIds size : " << nonZeroColIds.size() << std::endl; 
    if(!refineB2) nonZeroColIds.clear();
    for (auto dfid : nonZeroColIds)
    {
        auto contour            = extract_column_contour(sc.baseVertsPtr, spatial_verts, one_column_simp_opt, dfid);
        const int num_polyhedra = contour.get_num_polyhedra();
        const int num_vertices  = contour.get_num_vertices();

        std::vector<double>             contour_time;  contour_time.reserve(num_vertices);
        std::vector<int>                contour_index; contour_index.reserve(num_vertices);
        std::vector<Eigen::RowVector4d>  contour_pos;   contour_pos.reserve(num_vertices);
        parse_vertices(contour, contour_time, contour_index, contour_pos, spatial_verts);

        // Cache time lists to avoid repeated getTimeList_f() calls per poly loop
        const std::array<vertexCol::time_list_f, 4> time = {
            sc.baseVertsPtr[0]->getTimeList_f(), sc.baseVertsPtr[1]->getTimeList_f(),
            sc.baseVertsPtr[2]->getTimeList_f(), sc.baseVertsPtr[3]->getTimeList_f()};

        std::vector<mtetcol::Index> vert_id;
        vert_id.reserve(num_vertices);
        
        for (int pi = 0; pi < num_polyhedra; pi++) {
            const bool simple = contour.is_polyhedron_regular(pi);
            parse_polyhedron(contour, pi, vert_id);
            if (simple) assert(vert_id.size() == 4);
            for (size_t ci = 0; ci < cell5Col.size(); ci++) {
                if(isTemporalRefine && cell5TempUpdated[ci])
                {
                    continue;
                }
                if(sc.cellDFuncFt0XIds.row(ci)(dfid) == 0) continue;

    #if time_profile
                Timer find_intersect_timer(find_intersect, [&](auto t, auto ms) {
                    combine_timer(profileTimer, profileCount, t, ms); });
    #endif
                const bool intersect =
                    cell5_intersects_poly(cell5Col[ci], vert_id,
                                        contour_time, contour_index, time);
    #if time_profile
                find_intersect_timer.Stop();
    #endif
                if (!intersect) continue;
                if (simple) {
    #if time_profile
                    Timer second_func_timer(second_func, [&](auto t, auto ms) {
                        combine_timer(profileTimer, profileCount, t, ms); });
    #endif
                    
                    for (int vi = 0; vi < (int)vert_id.size(); vi++) {
                        sc.polygonVerts[vi].coord       = contour_pos[vert_id[vi]];
                        sc.polygonVerts[vi].valGradList = funcs[dfid](sc.polygonVerts[vi].coord); 
                        // sc.polygonVertsPtr[vi] = &sc.polygonVerts[vi];
                    }
                    if (refine3D(sc.polygonVerts, threshold))
                    {
                        try_push_space(spaceQ, baseSub, tid,
                                    sc.longest_edge, sc.longest_edge_length, min_tet_edge_length);
                        break;
                    }
                        
    #if time_profile
                    second_func_timer.Stop();
    #endif
                } else if (20.0 * sc.longest_edge_length > threshold) {
    #if time_profile
                    Timer non_simple_poly_timer(non_simple_poly, [&](auto t, auto ms) {
                        combine_timer(profileTimer, profileCount, t, ms); });
    #endif
                    try_push_space(spaceQ, baseSub, tid,
                                sc.longest_edge, sc.longest_edge_length, min_tet_edge_length);
                    break;
    #if time_profile
                    non_simple_poly_timer.Stop();
    #endif
                }
            }
        }
    }

    bool refineB4 = true; 
    if(!refineB4) pairToTets.clear();
    for (const auto& [pair, tets] : pairToTets)
    {
        auto [fid_a, fid_b] = pair;
        // check function pair {id_a, id_b} against tet tetId
        auto contour            = extract_column_contour(sc.baseVertsPtr, spatial_verts,
                                        one_column_simp_opt, fid_a, fid_b);
        const int num_polyhedra = contour.get_num_polyhedra();
        const int num_vertices  = contour.get_num_vertices();
        std::vector<double>             contour_time;  contour_time.reserve(num_vertices);
        std::vector<int>                contour_index; contour_index.reserve(num_vertices);
        std::vector<Eigen::RowVector4d>  contour_pos;   contour_pos.reserve(num_vertices);
        parse_vertices(contour, contour_time, contour_index, contour_pos, spatial_verts);

        const std::array<vertexCol::time_list_f, 4> time = {
        sc.baseVertsPtr[0]->getTimeList_f(), sc.baseVertsPtr[1]->getTimeList_f(),
        sc.baseVertsPtr[2]->getTimeList_f(), sc.baseVertsPtr[3]->getTimeList_f()};

        std::vector<mtetcol::Index> vert_id;
        vert_id.reserve(num_vertices);
        
        for (int pi = 0; pi < num_polyhedra; pi++) 
        {
            const bool simple = contour.is_polyhedron_regular(pi);
            parse_polyhedron(contour, pi, vert_id);
            if (simple) assert(vert_id.size() == 4);

            // for (size_t ci = 0; ci < cell5Col.size(); ci++) 
            for (size_t ci : tets) 
            {
    #if time_profile
                Timer find_intersect_timer(find_intersect, [&](auto t, auto ms) {
                    combine_timer(profileTimer, profileCount, t, ms); });
    #endif
                const bool intersect =
                    cell5_intersects_poly(cell5Col[ci], vert_id,
                                        contour_time, contour_index, time);
    #if time_profile
                find_intersect_timer.Stop();
    #endif
                if (!intersect) continue;
                if (simple) {
    #if time_profile
                    Timer second_func_timer(second_func, [&](auto t, auto ms) {
                        combine_timer(profileTimer, profileCount, t, ms); });
    #endif
                    
                    for (int vi = 0; vi < (int)vert_id.size(); vi++) {
                        sc.polygonVerts[vi].coord       = contour_pos[vert_id[vi]];
                        auto valGradListA = funcs[fid_a](sc.polygonVerts[vi].coord);
                        auto valGradListB = funcs[fid_a](sc.polygonVerts[vi].coord); 
                        sc.polygonVerts[vi].valGradList = {valGradListA.first + valGradListB.first, 
                        valGradListA.second + valGradListB.second}; 
                        // sc.polygonVertsPtr[vi] = &sc.polygonVerts[vi];
                    }

                    if (refine3D(sc.polygonVerts, threshold))
                    {
                        try_push_space(spaceQ, baseSub, tid,
                                    sc.longest_edge, sc.longest_edge_length, min_tet_edge_length);
                        break;
                    }
                        
    #if time_profile
                    second_func_timer.Stop();
    #endif
                } else if (20.0 * sc.longest_edge_length > threshold) {
    #if time_profile
                    Timer non_simple_poly_timer(non_simple_poly, [&](auto t, auto ms) {
                        combine_timer(profileTimer, profileCount, t, ms); });
    #endif
                    try_push_space(spaceQ, baseSub, tid,
                                sc.longest_edge, sc.longest_edge_length, min_tet_edge_length);
                    break;
    #if time_profile
                    non_simple_poly_timer.Stop();
    #endif
                }
            }
        }

        
    }

#if time_profile
    extract_first_iso_timer.Stop();
    Timer second_part_timer(second_part, [&](auto t, auto ms) {
        combine_timer(profileTimer, profileCount, t, ms); });
#endif

#endif // !only_stage1
}

// ============================================================
//  Thread-local context for parallel push_one_col evaluation
// ============================================================
struct ThreadLocalCtx {
    ColScratch scratch;
    std::vector<TimeElem>  timeQ;
    std::vector<SpaceElem> spaceQ;
    std::vector<uint64_t>  active_tet_keys;
    std::vector<vertex4d*> active_vert_ptrs;          // vertex4d to mark active
    std::vector<std::array<VertexId, 4>> inside_tets;
    double min_tet_ratio = 1.0;
};

// ============================================================
//  Thread-local push_one_col
//  Reads from shared ctx, writes only to ThreadLocalCtx.
// ============================================================
static void push_one_col_tl(mtet::TetId tid, PushOneColCtx& ctx, ThreadLocalCtx& tl)
{
    auto& grid       = ctx.grid;
    auto& vertexMap  = ctx.vertexMap;
    auto& funcs      = ctx.funcs;
    auto& sc         = tl.scratch;
    const double threshold            = ctx.threshold;
    const double traj_threshold       = ctx.traj_threshold;
    const int    insideness_check     = ctx.insideness_check;
    const double time_scale           = ctx.time_scale;
    const double min_tet_radius_ratio = ctx.min_tet_radius_ratio;
    const double min_tet_edge_length  = ctx.min_tet_edge_length;
    auto& profileTimer = ctx.profileTimer;
    auto& profileCount = ctx.profileCount;

    const size_t CSGFuncNum = funcs.size();
    if (!grid.has_tet(tid)) return;
    const auto& tetVids = grid.get_tet(tid);

    simpCol::cell5_list cell5Col;
    sampleCol(tetVids, vertexMap, cell5Col);

    gather_base_verts(tetVids, vertexMap, sc.baseVertsPtr, sc.baseCoord);

    // ── Tet quality check ────────────────────────────────────────────────
    {
        std::array<Eigen::RowVector3d, 4> tet_pts = {
            sc.baseCoord[0].head<3>(), sc.baseCoord[1].head<3>(),
            sc.baseCoord[2].head<3>(), sc.baseCoord[3].head<3>()
        };
        const auto tet_ratio = tet_radius_ratio(tet_pts);
        if (tet_ratio < min_tet_radius_ratio) {
            tl.inside_tets.push_back({tetVids[0], tetVids[1], tetVids[2], tetVids[3]});
            return;
        }
        tl.min_tet_ratio = std::min(tl.min_tet_ratio, tet_ratio);
    }

    find_longest_edge(grid, tid, sc.longest_edge, sc.longest_edge_length);

    // Helper: thread-local space push
    auto try_push_space_tl = [&]() -> bool {
        if (sc.longest_edge_length <= min_tet_edge_length) return false;
        tl.spaceQ.emplace_back(sc.longest_edge_length, tid, sc.longest_edge);
        std::push_heap(tl.spaceQ.begin(), tl.spaceQ.end(), compSpace);
        return true;
    };

    // ── Stage 1.1: silhouette refinement per cell5 ───────────────────────
    bool terminate    = false;
    bool baseSub      = false;
    bool activeCol    = false;

    sc.cellDomFuncIds.assign(cell5Col.size(), {});
    sc.cellDFuncFt0XIds.setZero(cell5Col.size(), CSGFuncNum);
    sc.cellDFunc0XIds.setZero(cell5Col.size(), CSGFuncNum);

    std::vector<Eigen::MatrixXi> equalSurfaceFuncPairs(
        CSGFuncNum, Eigen::MatrixXi::Zero(CSGFuncNum, CSGFuncNum));
    std::unordered_map<std::pair<size_t, size_t>, std::vector<size_t>, PairHash> pairToTets;

    size_t last_v_ind = 0;
    if (isTemporalRefine) {
        for (size_t tvi = 0; tvi < 4; ++tvi) {
            if (value_of(tetVids[tvi]) == tempRefineVtId) {
                last_v_ind = tvi;
            }
        }
    }

    std::vector<bool> cell5TempUpdated(cell5Col.size(), false);
    Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor> bezierValsShared;
    Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor> bezierFtValsShared;
    bezierValsShared.setZero(CSGFuncNum, 35);
    bezierFtValsShared.setZero(CSGFuncNum, 35);

    for (size_t ci = 0; ci < cell5Col.size(); ci++) {

        if (isTemporalRefine) {
            bool isUpdatedTempTet = false;
            if (cell5Col[ci].bot(last_v_ind) == tempRefineTimeVal ||
                (cell5Col[ci].top() == tempRefineTimeVal && cell5Col[ci].hash[4] == last_v_ind)) {
                isUpdatedTempTet = true;
                cell5TempUpdated[ci] = true;
            }
            if (!isUpdatedTempTet) continue;
        }

        bind_cell5_verts(cell5Col[ci], sc.baseVertsPtr, sc.tet4DVertsPtr);

        std::vector<size_t> domFIds;
        calBezierCoordsAndDomFuncIds(sc.tet4DVertsPtr, profileTimer, profileCount,
                                     bezierValsShared, bezierFtValsShared, domFIds);

        bool choice = false, zeroX = false;
        bool inside = false;
        bool needs_refine = refineFtCSG(
            sc.tet4DVertsPtr, bezierValsShared, bezierFtValsShared, domFIds,
            traj_threshold, choice, inside, zeroX,
            profileTimer, profileCount,
            sc.cellDFuncFt0XIds.row(ci), sc.cellDFunc0XIds.row(ci));

        sc.zeroX_list[ci] = zeroX;
        if (insideness_check && inside) {
            tl.inside_tets.push_back({tetVids[0], tetVids[1], tetVids[2], tetVids[3]});
            return;
        }
        if (zeroX) activeCol = true;

        bool eqaulSurf0X = false;
        bool refineB3 = true;

        if (!needs_refine && refineB3) {
            double ef_max_error = 0;
            // bool choice = false;
            for (size_t id_a = 0; id_a < CSGFuncNum; ++id_a) {
                if (sc.cellDFunc0XIds.row(ci)(id_a) == 0) continue;
                for (size_t id_b = id_a + 1; id_b < CSGFuncNum; ++id_b) {
                    if (sc.cellDFunc0XIds.row(ci)(id_b) == 0) continue;
                    needs_refine = refineEqualSurfaceCSG(
                        sc.tet4DVertsPtr, bezierValsShared, bezierFtValsShared,
                        traj_threshold, {id_a, id_b}, choice, eqaulSurf0X, ef_max_error,
                        profileTimer, profileCount);
                    if (eqaulSurf0X) {
                        activeCol = true;
                        pairToTets[{id_a, id_b}].push_back(ci);
                    }
                    if(needs_refine) break;
                }
                if(needs_refine) break;
            }
        }
        if (needs_refine) {
            sc.subList[ci]     = true;
            sc.timeLenList[ci] = sc.tet4DVertsPtr[0]->time - sc.tet4DVertsPtr[4]->time;
            sc.timeList[ci]    = (sc.tet4DVertsPtr[0]->time + sc.tet4DVertsPtr[4]->time) / 2;
            sc.indList[ci]     = cell5Col[ci].hash[4];
            sc.choiceList[ci]  = choice;
            sc.choiceList[ci]  = sc.timeLenList[ci] / double(MAX_TIME) * time_scale > sc.longest_edge_length;
        } else {
            sc.subList[ci] = false;
        }
    }

    // ── Push refinement requests to thread-local queues ──────────────────
    for (size_t ci = 0; ci < cell5Col.size(); ci++) {
        if (!sc.subList[ci]) continue;
        terminate = true;
        if (sc.choiceList[ci]) {
            if (sc.timeLenList[ci] > MIN_TIME) {

                // double dt_sum = 0; double dx_sum = 0; double dy_sum = 0; double dz_sum = 0; 
                // for(size_t fi = 0; fi < sc.cellDFunc0XIds.cols(); ++ fi)
                // {
                //     if(sc.cellDFunc0XIds(ci,fi) != 0)
                //     {
                //         bind_cell5_verts(cell5Col[ci], sc.baseVertsPtr, sc.tet4DVertsPtr);
                //         const auto& g1s = sc.tet4DVertsPtr[0]->grads;
                //         const auto& g2s = sc.tet4DVertsPtr[1]->grads;
                //         const auto& g3s = sc.tet4DVertsPtr[2]->grads;
                //         const auto& g4s = sc.tet4DVertsPtr[3]->grads;
                //         const auto& g5s = sc.tet4DVertsPtr[4]->grads;
                //         dt_sum += abs(g1s(fi,3)) + abs(g2s(fi,3)) + abs(g3s(fi,3)) + abs(g4s(fi,3)) + abs(g5s(fi,3));
                //         dx_sum += abs(g1s(fi,0)) + abs(g2s(fi,0)) + abs(g3s(fi,0)) + abs(g4s(fi,0)) + abs(g5s(fi,0));
                //         dy_sum += abs(g1s(fi,1)) + abs(g2s(fi,1)) + abs(g3s(fi,1)) + abs(g4s(fi,1)) + abs(g5s(fi,1));
                //         dz_sum += abs(g1s(fi,2)) + abs(g2s(fi,2)) + abs(g3s(fi,2)) + abs(g4s(fi,2)) + abs(g5s(fi,2));
                //     }
                // }
                // double local_time_scale = std::max(1.0, (dt_sum + 0.000001)/( (dx_sum + dy_sum + dz_sum)/ 3 + 0.000001));
                
                {
                    tl.timeQ.emplace_back(
                    (double)sc.timeLenList[ci] * time_scale / MAX_TIME,
                    tid, tetVids[sc.indList[ci]], (int)sc.timeList[ci]);
                    std::push_heap(tl.timeQ.begin(), tl.timeQ.end(), compTime);
                }
                
            }
        } else {
            if (!baseSub) {
                if (try_push_space_tl()) baseSub = true;
            }
        }
    }

    // ── Cap refinement at t=0 and t=1 boundaries ─────────────────────────
    bool refineCap = true;
    if (!baseSub && !terminate && refineCap &&
        (!isTemporalRefine || (isTemporalRefine && cell5TempUpdated[0])))
    {
        for (size_t dfId = 0; dfId < funcs.size(); ++dfId) {
            if (sc.cellDFunc0XIds.row(0)(dfId) == 0) continue;
            activeCol = true;
            for (size_t i = 0; i < 4; i++) {
                sc.capVertsPtr[i] = &sc.baseVertsPtr[i]->vert4dList.front();
            }
            if (refine3DCSG(sc.capVertsPtr, threshold, dfId)) {
                if (try_push_space_tl()) baseSub = true;
                break;
            }
        }
    }

    if (!baseSub && !terminate && refineCap &&
        (!isTemporalRefine || (isTemporalRefine && cell5TempUpdated[cell5Col.size() - 1])))
    {
        for (size_t dfId = 0; dfId < funcs.size(); ++dfId) {
            if (sc.cellDFunc0XIds.row(cell5Col.size() - 1)(dfId) == 0) continue;
            activeCol = true;
            for (size_t i = 0; i < 4; i++) {
                sc.capVertsPtr[i] = &sc.baseVertsPtr[i]->vert4dList.back();
            }
            if (refine3DCSG(sc.capVertsPtr, threshold, dfId)) {
                if (try_push_space_tl()) baseSub = true;
                break;
            }
        }
    }

    // ── Record activity (deferred to serial merge) ───────────────────────
    if (activeCol) {
        tl.active_tet_keys.push_back(getTetKeyByVids(tetVids));
        for (size_t vid = 0; vid < 5; vid++) {
            tl.active_vert_ptrs.push_back(sc.tet4DVertsPtr[vid]);
        }
    }

    if (baseSub) terminate = true;
    if (terminate) return;

    // ── Stage 2: contour-based refinement ────────────────────────────────
#ifndef only_stage1
    std::array<mtet::Scalar, 12> spatial_verts;
    collect_spatial_verts(grid, tetVids, spatial_verts);

    bool refineB2 = true;
    std::vector<int> nonZeroColIds;
    for (int j = 0; j < sc.cellDFuncFt0XIds.cols(); ++j)
        if (sc.cellDFuncFt0XIds.col(j).any())
            nonZeroColIds.push_back(j);
    if (!refineB2) nonZeroColIds.clear();

    for (auto dfid : nonZeroColIds) {
        auto contour = extract_column_contour(
            sc.baseVertsPtr, spatial_verts, one_column_simp_opt, dfid);
        const int num_polyhedra = contour.get_num_polyhedra();
        const int num_vertices  = contour.get_num_vertices();

        std::vector<double>             contour_time;  contour_time.reserve(num_vertices);
        std::vector<int>                contour_index; contour_index.reserve(num_vertices);
        std::vector<Eigen::RowVector4d> contour_pos;   contour_pos.reserve(num_vertices);
        parse_vertices(contour, contour_time, contour_index, contour_pos, spatial_verts);

        const std::array<vertexCol::time_list_f, 4> time = {
            sc.baseVertsPtr[0]->getTimeList_f(), sc.baseVertsPtr[1]->getTimeList_f(),
            sc.baseVertsPtr[2]->getTimeList_f(), sc.baseVertsPtr[3]->getTimeList_f()};

        std::vector<mtetcol::Index> vert_id;
        vert_id.reserve(num_vertices);

        bool space_pushed = false;
        for (int pi = 0; pi < num_polyhedra && !space_pushed; pi++) {
            const bool simple = contour.is_polyhedron_regular(pi);
            parse_polyhedron(contour, pi, vert_id);
            if (simple) assert(vert_id.size() == 4);

            for (size_t ci = 0; ci < cell5Col.size(); ci++) {
                if (isTemporalRefine && cell5TempUpdated[ci]) continue;
                if (sc.cellDFuncFt0XIds.row(ci)(dfid) == 0) continue;

                const bool intersect = cell5_intersects_poly(
                    cell5Col[ci], vert_id, contour_time, contour_index, time);
                if (!intersect) continue;

                if (simple) {
                    for (int vi = 0; vi < (int)vert_id.size(); vi++) {
                        sc.polygonVerts[vi].coord       = contour_pos[vert_id[vi]];
                        sc.polygonVerts[vi].valGradList = funcs[dfid](sc.polygonVerts[vi].coord);
                    }
                    if (refine3D(sc.polygonVerts, threshold)) {
                        if (try_push_space_tl()) { baseSub = true; space_pushed = true; }
                        break;
                    }
                } else if (20.0 * sc.longest_edge_length > threshold) {
                    if (try_push_space_tl()) { baseSub = true; space_pushed = true; }
                    break;
                }
            }
        }
        if (space_pushed) break;
    }

    bool refineB4 = true;
    if (!refineB4) pairToTets.clear();
    for (const auto& [pair, tets] : pairToTets) {
        if (baseSub) break;
        auto [fid_a, fid_b] = pair;

        auto contour = extract_column_contour(
            sc.baseVertsPtr, spatial_verts, one_column_simp_opt, fid_a, fid_b);
        const int num_polyhedra = contour.get_num_polyhedra();
        const int num_vertices  = contour.get_num_vertices();
        std::vector<double>             contour_time;  contour_time.reserve(num_vertices);
        std::vector<int>                contour_index; contour_index.reserve(num_vertices);
        std::vector<Eigen::RowVector4d> contour_pos;   contour_pos.reserve(num_vertices);
        parse_vertices(contour, contour_time, contour_index, contour_pos, spatial_verts);

        const std::array<vertexCol::time_list_f, 4> time = {
            sc.baseVertsPtr[0]->getTimeList_f(), sc.baseVertsPtr[1]->getTimeList_f(),
            sc.baseVertsPtr[2]->getTimeList_f(), sc.baseVertsPtr[3]->getTimeList_f()};

        std::vector<mtetcol::Index> vert_id;
        vert_id.reserve(num_vertices);

        bool space_pushed = false;
        for (int pi = 0; pi < num_polyhedra && !space_pushed; pi++) {
            const bool simple = contour.is_polyhedron_regular(pi);
            parse_polyhedron(contour, pi, vert_id);
            if (simple) assert(vert_id.size() == 4);

            for (size_t ci : tets) {
                const bool intersect = cell5_intersects_poly(
                    cell5Col[ci], vert_id, contour_time, contour_index, time);
                if (!intersect) continue;

                if (simple) {
                    for (int vi = 0; vi < (int)vert_id.size(); vi++) {
                        sc.polygonVerts[vi].coord = contour_pos[vert_id[vi]];
                        auto valGradListA = funcs[fid_a](sc.polygonVerts[vi].coord);
                        auto valGradListB = funcs[fid_a](sc.polygonVerts[vi].coord);  // NOTE: original has fid_a here too
                        sc.polygonVerts[vi].valGradList = {
                            valGradListA.first + valGradListB.first,
                            valGradListA.second + valGradListB.second};
                    }
                    if (refine3D(sc.polygonVerts, threshold)) {
                        if (try_push_space_tl()) { baseSub = true; space_pushed = true; }
                        break;
                    }
                } else if (20.0 * sc.longest_edge_length > threshold) {
                    if (try_push_space_tl()) { baseSub = true; space_pushed = true; }
                    break;
                }
            }
        }
    }
#endif // !only_stage1
}


std::vector<mtet::TetId> candidataTetIds_SM;

// ============================================================
//  Temporal subdivision step
// ============================================================

static void do_temporal_split(
    const TimeElem& time_ele,
    mtet::MTetMesh& grid,
    vertExtrude&    vertexMap,
    insidenessMap&  insideMap,
    const CSGFuncs& funcs,
    int&            splits,
    int&            temporal_splits,
    PushOneColCtx&  ctx)
{
    auto [_, tid, vid, time] = time_ele;

    if (!grid.has_tet(tid)) return;
    const std::span<VertexId, 4> vs = grid.get_tet(tid);
    if (vertexMap[value_of(vid)].timeExist[time] || insideMap[vs]) return;
    // if (vertexMap[value_of(vid)].timeExist[time] ) return;

    vertexCol timeList = vertexMap[value_of(vid)];
    splits++;
    temporal_splits++;
    timeList.timeExist[time] = true;

    // Prune stale tet associations
    vertexCol::vertToSimp live;
    live.reserve(timeList.vertTetAssoc.size());
    for (auto t : timeList.vertTetAssoc)
        if (grid.has_tet(t)) live.emplace_back(t);
    timeList.vertTetAssoc = std::move(live);

    // Insert new 4D sample
    vertex4d newVert(funcs.size());
    newVert.time = time;
    // double tfp = (double)time / MAX_TIME * (time_end_G - time_start_G) + time_start_G;
    double tfp = (double)time / MAX_TIME;

    newVert.coord = {
        timeList.vert4dList[0].coord[0],
        timeList.vert4dList[0].coord[1],
        timeList.vert4dList[0].coord[2],
        tfp};
        
    for(size_t dfi = 0; dfi < funcs.size(); ++dfi)
    {
        auto cur_val_grad = funcs[dfi](newVert.coord);
        newVert.vals[dfi] = cur_val_grad.first;
        newVert.grads.row(dfi) = cur_val_grad.second;    
    }
    newVert.valGradList = {newVert.vals[0], newVert.grads.row(0)}; 
    newVert.active = false;
    // newVert.valGradList = func(newVert.coord);
    timeList.insertTime(newVert);
    vertexMap[value_of(vid)] = std::move(timeList);

    isTemporalRefine = false;
    tempRefineVtId = value_of(vid);
    tempRefineTimeVal = time;
    for (auto assocTet : vertexMap[value_of(vid)].vertTetAssoc) {
#if time_profile
        Timer ref_crit_timer(ref_crit, [&](auto t, auto ms) {
            combine_timer(ctx.profileTimer, ctx.profileCount, t, ms); });
#endif  
        if(!refine_using_openmp) push_one_col(assocTet, ctx); 
        candidataTetIds_SM.push_back(assocTet);
#if time_profile
        ref_crit_timer.Stop();
#endif
    }

    isTemporalRefine = false;

}

// ============================================================
//  Spatial subdivision step
// ============================================================

static void do_spatial_split(
    const SpaceElem& space_ele,
    mtet::MTetMesh&  grid,
    vertExtrude&     vertexMap,
    insidenessMap&  insideMap,
    const CSGFuncs&  funcs,
    int&             splits,
    int&             spatial_splits,
    PushOneColCtx&   ctx)
{
    auto [space_edge_len, tid, eid] = space_ele;
    if (!grid.has_tet(tid)) return;

    std::span<VertexId, 4> vs = grid.get_tet(tid);
    if (insideMap[vs]) {
        return;
    }
    splits++;
    spatial_splits++;
    const std::array<VertexId, 2> vs_old = grid.get_edge_vertices(eid);
    //  std::cout << "start to split edge" << std::endl;
    auto [vid, eid0, eid1] = grid.split_edge(eid);
    // std::cout << " split edge successfully " << std::endl;

    vertexMap[value_of(vid)] =
        make_new_spatial_vert(grid, vid,
                              vertexMap[value_of(vs_old[0])],
                              vertexMap[value_of(vs_old[1])], funcs);

    auto process_edge = [&](mtet::EdgeId e) {
        grid.foreach_tet_around_edge(e, [&](mtet::TetId t) {
            std::span<VertexId, 4> tvs = grid.get_tet(t);
            for (size_t i = 0; i < 4; i++)
                vertexMap[value_of(tvs[i])].vertTetAssoc.emplace_back(t);
#if time_profile
            Timer ref_crit_timer(ref_crit, [&](auto timer, auto ms) {
                combine_timer(ctx.profileTimer, ctx.profileCount, timer, ms); });
#endif
            if(!refine_using_openmp) push_one_col(t, ctx);
            candidataTetIds_SM.push_back(t);
#if time_profile
            ref_crit_timer.Stop();
#endif
        });
    };

    process_edge(eid0);
    process_edge(eid1);
}

// ============================================================
//  gridRefine  — now just orchestration
// ============================================================

bool gridRefineCSG(
    mtet::MTetMesh& grid,
    vertExtrude& vertexMap,
    insidenessMap& insideMap,
    const CSGFuncs& funcs,
    CSGFunction csg_f,
    const double threshold,
    const double traj_threshold,
    const int max_splits,
    const int insideness_check,
    std::array<double, timer_amount>& profileTimer,
    std::array<size_t, timer_amount>& profileCount,
    std::unordered_map<uint64_t, int>& colActiveMap,
    size_t initial_time_samples,
    double min_tet_radius_ratio,
    double min_tet_edge_length,
    const std::string& out_dir)
{

    set_csg_val_func(csg_f);
    // initial_time_samples = 16;
    init5CGridCSG(initial_time_samples, grid, funcs, MAX_TIME, vertexMap);

    const double time_scale = calTimeGlobalScaleWithInitGridCSG(vertexMap) ;
    std::cout << " --- time scale : " << time_scale << std::endl;
    // init shared data 
    std::string log_path = out_dir + "/run_log.txt";
    std::ofstream log_file(log_path, std::ios::app);
    auto file_log = [&](const std::string& msg) {
        // std::cout << msg << std::endl;
        if (log_file.is_open()) log_file << msg << std::endl;
    };
    file_log(" --- time scale : " + std::to_string(time_scale)); 
    colActiveMapPtr = &colActiveMap;
    // Queues
    std::vector<TimeElem>  timeQ;
    std::vector<SpaceElem> spaceQ;
    // Shared scratch (one allocation for the whole run)
    ColScratch scratch;
    scratch.polygonVerts.fill(vertex4d(funcs.size()));
    // scratch.polygonVerts.fill(vertex4d(4));
    
    int splits = 0, temporal_splits = 0, spatial_splits = 0;
    double min_tet_ratio = 1.0;

    PushOneColCtx ctx{
        grid, vertexMap, insideMap, funcs,
        threshold, traj_threshold, insideness_check,
        time_scale, min_tet_radius_ratio, min_tet_edge_length,
        timeQ, spaceQ, profileTimer, profileCount,
        min_tet_ratio, scratch};

    // Initial pass
    grid.seq_foreach_tet(
        [&](mtet::TetId tid, [[maybe_unused]] std::span<const mtet::VertexId, 4>) {
            push_one_col(tid, ctx);
            candidataTetIds_SM.push_back(tid);
        });

    // Main refinement loop
    TimeElem  time_ele{};
    SpaceElem space_ele{};
    bool has_time_ele  = false;
    bool has_space_ele = false;
    size_t tempRefineCount = 0;
    size_t spatialRefineCount = 0;


    // std::cout << " start refineFtCSG  loop " << std::endl;
    while ((!timeQ.empty() || !spaceQ.empty()) && splits < max_splits) {

        bool refine_temporal = false;
        if (!timeQ.empty() && !has_time_ele) {
            std::pop_heap(timeQ.begin(), timeQ.end(), compTime);
            time_ele = timeQ.back(); timeQ.pop_back();
            has_time_ele = true;
            // refine_temporal = true;
        }
        if (!spaceQ.empty() && !has_space_ele) {
            std::pop_heap(spaceQ.begin(), spaceQ.end(), compSpace);
            space_ele = spaceQ.back(); spaceQ.pop_back();
            has_space_ele = true;
        }

        refine_temporal =
            has_time_ele && (!has_space_ele ||
                std::get<0>(time_ele) > std::get<0>(space_ele));

        if (refine_temporal) {
            has_time_ele = false;
            do_temporal_split(time_ele, grid, vertexMap, insideMap, funcs,
                              splits, temporal_splits, ctx);
        } else {
            has_space_ele = false;
            do_spatial_split(space_ele, grid, vertexMap, insideMap, funcs,
                             splits, spatial_splits, ctx);
        }
    }

    size_t valid_tet_count = 0;

    for(auto& ele : colActiveMap)
    {
        if(ele.second)
        {
            valid_tet_count ++;
        }
    }
    std::cout << " --------- temporal refine count :  "<< temporal_splits << std::endl;
    std::cout << " --------- spatial refine count :  "<< spatial_splits << std::endl;
    file_log(" --------- temporal refine count :  " + std::to_string(temporal_splits)); 
    file_log(" --------- spatial refine count : " + std::to_string(spatial_splits)); 
    log_file.close();
    // std::cout << " --------- active_3d_tet_count "<< valid_tet_count << std::endl;
    // std::cout << " --------- 3d_tet_count "<<  << std::endl;
    sweep::logger().info(
        "Total splits: {}  Spatial splits: {}  Minimum tet radius ratio: {}",
        splits, spatial_splits, min_tet_ratio);
    return true;
}

bool gridRefineCSGParallel(
    mtet::MTetMesh& grid,
    vertExtrude& vertexMap,
    insidenessMap& insideMap,
    const CSGFuncs& funcs,
    CSGFunction csg_f,
    const double threshold,
    const double traj_threshold,
    const int max_splits,
    const int insideness_check,
    std::array<double, timer_amount>& profileTimer,
    std::array<size_t, timer_amount>& profileCount,
    std::unordered_map<uint64_t, int>& colActiveMap,
    size_t initial_time_samples,
    double min_tet_radius_ratio,
    double min_tet_edge_length,
    const std::string& out_dir,
    double set_time_start,
    double set_time_end)
{
    time_end_G = set_time_end;
    time_start_G = set_time_start;
    set_csg_val_func(csg_f);
    init5CGridCSG(initial_time_samples, grid, funcs, MAX_TIME, vertexMap);


    std::string log_path = out_dir + "/run_log.txt";
    std::ofstream log_file(log_path, std::ios::app);
    auto file_log = [&](const std::string& msg) {
        if (log_file.is_open()) log_file << msg << std::endl;
    };

    const double time_scale = calTimeGlobalScaleWithInitGridCSG(vertexMap);
    std::cout << " --- time scale : " << time_scale << std::endl;

    file_log(" --- time scale : " + std::to_string(time_scale)); 

    colActiveMapPtr = &colActiveMap;

    // Global queues (merged from per-thread)
    std::vector<TimeElem>  timeQ;
    std::vector<SpaceElem> spaceQ;

    int splits = 0, temporal_splits = 0, spatial_splits = 0;
    double min_tet_ratio = 1.0;

    // Single shared scratch for serial split functions only
    ColScratch serial_scratch;
    serial_scratch.polygonVerts.fill(vertex4d(funcs.size()));

    PushOneColCtx ctx{
        grid, vertexMap, insideMap, funcs,
        threshold, traj_threshold, insideness_check,
        time_scale, min_tet_radius_ratio, min_tet_edge_length,
        timeQ, spaceQ, profileTimer, profileCount,
        min_tet_ratio, serial_scratch};

    // ── Initial pass: collect all tets ───────────────────────────────────
    grid.seq_foreach_tet(
        [&](mtet::TetId tid, [[maybe_unused]] std::span<const mtet::VertexId, 4>) {
            candidataTetIds_SM.push_back(tid);
        });

    TimeElem  time_ele{};
    SpaceElem space_ele{};
    bool has_time_ele  = false;
    bool has_space_ele = false;
    size_t tempRefineCount = 0;
    size_t spatialRefineCount = 0;
    refine_using_openmp = true;
    // std::cout << " start to loop refine "<< std::endl;
    while (!candidataTetIds_SM.empty()) {
        const int n = (int)candidataTetIds_SM.size();
        const int num_threads = omp_get_max_threads();

        // ── Per-thread state ─────────────────────────────────────────────
        std::vector<ThreadLocalCtx> tls(num_threads);
        for (auto& tl : tls) {
            tl.scratch.polygonVerts.fill(vertex4d(funcs.size()));
        }

        // ── Parallel evaluation phase ────────────────────────────────────
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& tl = tls[tid];

            #pragma omp for schedule(dynamic, 16)
            for (int i = 0; i < n; ++i) {
                auto curTid = candidataTetIds_SM[i];
                push_one_col_tl(curTid, ctx, tl);
            }
        }
        // std::cout << " start to merege queues " << std::endl;

        // ── Serial merge phase ───────────────────────────────────────────
        for (auto& tl : tls) {
            // Merge time queue
            for (auto& e : tl.timeQ) {
                timeQ.push_back(e);
                std::push_heap(timeQ.begin(), timeQ.end(), compTime);
            }
            // Merge space queue
            for (auto& e : tl.spaceQ) {
                spaceQ.push_back(e);
                std::push_heap(spaceQ.begin(), spaceQ.end(), compSpace);
            }
            // Mark inside tets
            for (auto& vids : tl.inside_tets) {
                std::array<VertexId, 4> arr = vids;
                std::span<VertexId, 4> sp{arr.data(), 4};
                insideMap[sp] = true;
            }
            // Mark active tets
            for (auto k : tl.active_tet_keys) {
                (*colActiveMapPtr)[k] = 1;
            }
            // Mark active vertices
            for (auto* vp : tl.active_vert_ptrs) {
                vp->active = true;
            }
            // Reduce min tet ratio
            min_tet_ratio = std::min(min_tet_ratio, tl.min_tet_ratio);
        }

        candidataTetIds_SM.clear();

        // ── Serial mesh-mutation phase ───────────────────────────────────
        // Pop from queues and split. Splits are sequential because they
        // mutate grid topology. Each split appends new tets to
        // candidataTetIds_SM, processed in the next outer iteration.
        while ((!timeQ.empty() || !spaceQ.empty()) && splits < max_splits) {
            bool refine_temporal = false;

            if (!timeQ.empty() && !has_time_ele) {
                std::pop_heap(timeQ.begin(), timeQ.end(), compTime);
                time_ele = timeQ.back();
                timeQ.pop_back();
                has_time_ele = true;
                // refine_temporal = true;
            }
            if (!spaceQ.empty() && !has_space_ele) {
                std::pop_heap(spaceQ.begin(), spaceQ.end(), compSpace);
                space_ele = spaceQ.back();
                spaceQ.pop_back();
                has_space_ele = true;
            }

            refine_temporal =
                has_time_ele && (!has_space_ele ||
                                 std::get<0>(time_ele) > std::get<0>(space_ele));

            if (refine_temporal) {
                has_time_ele = false;
                do_temporal_split(time_ele, grid, vertexMap, insideMap, funcs,
                                  splits, temporal_splits, ctx);
            } else {
                has_space_ele = false;
                do_spatial_split(space_ele, grid, vertexMap, insideMap, funcs,
                                 splits, spatial_splits, ctx);
            }
        }
    }

    // ── Stats ────────────────────────────────────────────────────────────
    size_t valid_tet_count = 0;
    for (auto& ele : colActiveMap) {
        if (ele.second) valid_tet_count++;
    }
    std::cout << " --------- temporal refine count :  " << temporal_splits << std::endl;
    std::cout << " --------- spatial refine count :  "  << spatial_splits << std::endl;
    file_log(" --------- temporal refine count :  " + std::to_string(temporal_splits)); 
    file_log(" --------- spatial refine count : " + std::to_string(spatial_splits)); 
    log_file.close();
    sweep::logger().info(
        "Total splits: {}  Spatial splits: {}  Minimum tet radius ratio: {}",
        splits, spatial_splits, min_tet_ratio);
    return true;
}