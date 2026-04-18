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
#include <unordered_set>
#include "bezier_simplex.h"

#define parallel_bezier 0
//
//  col_gridgen.cpp  —  gridRefine refactor only
//  All other functions (sampleCol, resampleTimeCol, init5CGrid, …) are unchanged.
//

// ============================================================
//  Types & comparators (file-scope so they're shared by all helpers)
// ============================================================

using TimeElem  = std::tuple<mtet::Scalar, mtet::TetId, mtet::VertexId, int>;
using SpaceElem = std::tuple<mtet::Scalar, mtet::TetId, mtet::EdgeId>;
using CSGFuncs = std::vector<std::function<std::pair<Scalar, Eigen::RowVector4d>(Eigen::RowVector4d)>>;  
std::vector<uint32_t> one_column_simp_opt = {0, 1, 2, 3};

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


// shared data 
Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor> bezierValsShared;
Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor> bezierFtValsShared;
std::unordered_map<uint64_t, int>* colActiveMapPtr = nullptr;

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
    vertexCol::time_list  tSamples;
    std::set_union(v0Time.begin(), v0Time.end(),
                   v1Time.begin(), v1Time.end(),
                   std::back_inserter(tSamples));

    vertexCol::vert4d_list vertColList(tSamples.size());
    for (size_t i = 0; i < tSamples.size(); i++) {
        vertex4d vert(funcs.size());
        vert.time        = tSamples[i];
        const double tfp = (double)vert.time / MAX_TIME;
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
    const auto& tetVids = grid.get_tet(tid);

    simpCol::cell5_list cell5Col;
    sampleCol(tetVids, vertexMap, cell5Col);

    gather_base_verts(tetVids, vertexMap, sc.baseVertsPtr, sc.baseCoord);

    // Tet quality — mark degenerate tets as inside and skip further work
    {
        std::valarray<double> p0(sc.baseCoord[0].data(), 3);
        std::valarray<double> p1(sc.baseCoord[1].data(), 3);
        std::valarray<double> p2(sc.baseCoord[2].data(), 3);
        std::valarray<double> p3(sc.baseCoord[3].data(), 3);
        const auto tet_ratio = tet_radius_ratio({p0, p1, p2, p3});
        if (tet_ratio < min_tet_radius_ratio) {
            // insideMap[vs] = true;
            return;
        }
        min_tet_ratio = std::min(min_tet_ratio, tet_ratio);
    }

    find_longest_edge(grid, tid, sc.longest_edge, sc.longest_edge_length);

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

    // bezierValsShared.setZero(CSGFuncNum, 35);
    // bezierValsShared.setZero();
    for (size_t ci = 0; ci < cell5Col.size(); ci++) {
        bind_cell5_verts(cell5Col[ci], sc.baseVertsPtr, sc.tet4DVertsPtr);
        bezierValsShared.setZero();
        bezierFtValsShared.setZero();
        std::vector<size_t> domFIds;
        calBezierCoordsAndDomFuncIds(sc.tet4DVertsPtr, profileTimer, profileCount, bezierValsShared, domFIds);
            
        bool choice = false, zeroX = false;
        bool needs_refine  =
            // refineFtCSG(sc.tet4DVertsPtr, traj_threshold, choice, zeroX,
            //          profileTimer, profileCount, sc.cellDomFuncIds[ci]);
            // refineFtCSG(sc.tet4DVertsPtr, traj_threshold, choice, zeroX,
            //          profileTimer, profileCount, sc.cellDFuncFt0XIds.row(ci), sc.cellDFunc0XIds.row(ci));
            refineFtCSG(sc.tet4DVertsPtr, bezierValsShared, domFIds, traj_threshold, choice, zeroX, bezierFtValsShared,
                     profileTimer, profileCount, sc.cellDFuncFt0XIds.row(ci), sc.cellDFunc0XIds.row(ci));
        // sc.zeroX_list[ci] = zeroX;
        if(zeroX) activeCol = true;
        bool eqaulSurf0X = false;
        bool refineB3 = false;
        if(!needs_refine && refineB3)
        {
            for (size_t id_a = 0; id_a < CSGFuncNum; ++id_a)
            {
                if(sc.cellDFunc0XIds.row(ci)(id_a) == 0) continue;
                for (size_t id_b = id_a + 1; id_b < CSGFuncNum; ++id_b)
                {
                    if(sc.cellDFunc0XIds.row(ci)(id_b) == 0) continue;
                    needs_refine = refineEqualSurfaceCSG(sc.tet4DVertsPtr, bezierValsShared,bezierFtValsShared,
                    traj_threshold, {id_a, id_b}, choice, eqaulSurf0X,profileTimer,profileCount);
                    if(eqaulSurf0X) 
                    {
                        activeCol = true;
                        // equalSurfaceFuncPairs[ci](id_a, id_b) = 1;
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
            break;
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
        if (sc.choiceList[ci]) {
            if (sc.timeLenList[ci] > MIN_TIME) {
                timeQ.emplace_back(
                    (double)sc.timeLenList[ci] * time_scale / MAX_TIME,
                    tid, tetVids[sc.indList[ci]], (int)sc.timeList[ci]);
                std::push_heap(timeQ.begin(), timeQ.end(), compTime);
            }
        } else {
            if(!baseSub)
            {
                try_push_space(spaceQ, baseSub, tid,
                           sc.longest_edge, sc.longest_edge_length, min_tet_edge_length);
            }
        }
    }
    if(activeCol)
    {
        auto tetKey = getTetKeyByVids(tetVids);
        (*colActiveMapPtr)[tetKey] = 1;
    }
    // if(activeCol) set_base_verts_active(tetVids, vertexMap);
    // if((*colActiveMapPtr).find(value_of(tid)))
    // (*colActiveMapPtr)[value_of(tid)] = activeCol? 1: 0;

#if time_profile
    first_part_setup_timer2.Stop();
    Timer compute_caps_timer(compute_caps, [&](auto t, auto ms) {
        combine_timer(profileTimer, profileCount, t, ms); });
#endif
    // terminate = true;
    bool refineCap = false;
    if (!baseSub && !terminate && refineCap) {
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
    
    if (!baseSub && !terminate && refineCap) {
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
    // if(!activeCol)
    // {
    //     (*colActiveMapPtr)[value_of(tid)] = baseSub;
    // }
#if time_profile
    compute_caps_timer.Stop();
    first_part_timer.Stop();
#endif

    if(baseSub) terminate = true;

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

    // std::unordered_set<size_t> unique_domfIds;
    // for (const auto& vec : sc.cellDomFuncIds) {
    //     unique_domfIds.insert(vec.begin(), vec.end());
    // }
    bool refineB2 = false;
    std::vector<int> nonZeroColIds;
    for (int j = 0; j < sc.cellDFuncFt0XIds.cols(); ++j)
        if (sc.cellDFuncFt0XIds.col(j).any())
            nonZeroColIds.push_back(j);
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
                // if(sc.cellDomFuncIds[ci].find(dfid) == sc.cellDomFuncIds[ci].end())
                // {
                //     continue;
                // }
                if (!sc.zeroX_list[ci]) continue;
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

    bool refineB4 = false; 
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
//  Temporal subdivision step
// ============================================================

static void do_temporal_split(
    const TimeElem& time_ele,
    mtet::MTetMesh& grid,
    vertExtrude&    vertexMap,
    // insidenessMap&  insideMap,
    const CSGFuncs& funcs,
    int&            splits,
    int&            temporal_splits,
    PushOneColCtx&  ctx)
{
    auto [_, tid, vid, time] = time_ele;

    if (!grid.has_tet(tid)) return;
    if (vertexMap[value_of(vid)].timeExist[time]) return;

    // const std::span<VertexId, 4> vs = grid.get_tet(tid);
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
    newVert.coord = {
        timeList.vert4dList[0].coord[0],
        timeList.vert4dList[0].coord[1],
        timeList.vert4dList[0].coord[2],
        (double)time / MAX_TIME};
        
    for(size_t dfi = 0; dfi < funcs.size(); ++dfi)
    {
        auto cur_val_grad = funcs[dfi](newVert.coord);
        newVert.vals[dfi] = cur_val_grad.first;
        newVert.grads.row(dfi) = cur_val_grad.second;    
    }
    newVert.valGradList = {newVert.vals[0], newVert.grads.row(0)}; 
    // newVert.valGradList = func(newVert.coord);
    timeList.insertTime(newVert);
    vertexMap[value_of(vid)] = std::move(timeList);

    for (auto assocTet : vertexMap[value_of(vid)].vertTetAssoc) {
#if time_profile
        Timer ref_crit_timer(ref_crit, [&](auto t, auto ms) {
            combine_timer(ctx.profileTimer, ctx.profileCount, t, ms); });
#endif
        push_one_col(assocTet, ctx);
#if time_profile
        ref_crit_timer.Stop();
#endif
    }
}

// ============================================================
//  Spatial subdivision step
// ============================================================

static void do_spatial_split(
    const SpaceElem& space_ele,
    mtet::MTetMesh&  grid,
    vertExtrude&     vertexMap,
    const CSGFuncs&  funcs,
    int&             splits,
    int&             spatial_splits,
    PushOneColCtx&   ctx)
{
    auto [space_edge_len, tid, eid] = space_ele;
    if (!grid.has_tet(tid)) return;

    splits++;
    spatial_splits++;

    const std::array<VertexId, 2> vs_old = grid.get_edge_vertices(eid);
    auto [vid, eid0, eid1] = grid.split_edge(eid);

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
            push_one_col(t, ctx);
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

bool gridRefine2(
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
    double min_tet_edge_length)
{

    set_csg_val_func(csg_f);

    init5CGridCSG(initial_time_samples, grid, funcs, MAX_TIME, vertexMap);

    const double time_scale = calTimeGlobalScaleWithInitGridCSG(vertexMap);
    std::cout << " --- time scale : " << time_scale << std::endl;
    // init shared data 
    bezierValsShared.setZero(funcs.size(), 35);
    bezierFtValsShared.setZero(funcs.size(), 35);
    colActiveMapPtr = &colActiveMap;
    // Queues
    std::vector<TimeElem>  timeQ;
    std::vector<SpaceElem> spaceQ;
    // Shared scratch (one allocation for the whole run)
    ColScratch scratch;
    scratch.polygonVerts.fill(vertex4d(funcs.size()));
    
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
        });

    // Main refinement loop
    TimeElem  time_ele{};
    SpaceElem space_ele{};
    bool has_time_ele  = false;
    bool has_space_ele = false;

    while ((!timeQ.empty() || !spaceQ.empty()) && splits < max_splits) {

        if (!timeQ.empty() && !has_time_ele) {
            std::pop_heap(timeQ.begin(), timeQ.end(), compTime);
            time_ele = timeQ.back(); timeQ.pop_back();
            has_time_ele = true;
        }
        if (!spaceQ.empty() && !has_space_ele) {
            std::pop_heap(spaceQ.begin(), spaceQ.end(), compSpace);
            space_ele = spaceQ.back(); spaceQ.pop_back();
            has_space_ele = true;
        }

        const bool refine_temporal =
            has_time_ele && (!has_space_ele ||
                std::get<0>(time_ele) > std::get<0>(space_ele));

        if (refine_temporal) {
            has_time_ele = false;
            do_temporal_split(time_ele, grid, vertexMap, funcs,
                              splits, temporal_splits, ctx);
        } else {
            has_space_ele = false;
            do_spatial_split(space_ele, grid, vertexMap, funcs,
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

    std::cout << " --------- valid_tet_count "<< valid_tet_count << std::endl;

    sweep::logger().info(
        "Total splits: {}  Spatial splits: {}  Minimum tet radius ratio: {}",
        splits, spatial_splits, min_tet_ratio);
    return true;
}