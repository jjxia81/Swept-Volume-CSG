//
//  ref_crit.h
//  swept_volume
//
//  Created by Yiwen Ju on 12/10/24.
//

#ifndef ref_crit_h
#define ref_crit_h

#include <cmath>
#include <unordered_set>
#include "adaptive_column_grid.h"
#include "timer.h"



struct RefineResInfo{
    bool zeroX = false;
    bool isCap = false;
    bool fZeroX = false;
    bool chooseTemporalRefine = false;
    bool meetThreshold = false;
    double error = 0;
};

struct RefineTetGeoData{
    bool hasInit = false;
    double gradNorm;
    double determinant;
    void init(const std::array<vertex4d*, 5>& verts);
};


/// 4D (un-projected) distance check for the time derivative function for a given 4D 5-cell. The default threshold here is 0.01, and it will not change with user's input
/// @param[in] verts         An array of 5 4D vertices. This vertex4d data structure also stores this vertex value and gradient.
/// @param[in] threshold            Threshold for the maxmimum allowed distance error for this 5-cell not to be subdivided.
/// @param[out] inside          An tag that checks if this 5-cell is completely covered by the sweep. The check is conservative that it is only true if the bezier ordiantes of either top face or the bottom face of the 5-cell is completely inside(negative).
/// @param[out] choice         A tag that outputs the choice of subdivision for this 5-cell. It compares the distance error between bezier control points on the temporal edge with points on the top and bottom faces.
/// @param[out] zeroX           A tag that shows if this 5-cell contains the intersection of time derivative function and the sweep function.
/// @param[out] profileTimer            The time profiler. Details can be found in `timer.h`
/// @return         A boolean that decides if this 5-cell needs to be subdivided.
bool refineFt(
    const std::array<vertex4d*, 5>& verts,
    const double threshold,
    bool& inside,
    bool& choice,
    bool& zeroX,
    std::array<double, timer_amount>& profileTimer,
    std::array<size_t, timer_amount>& profileCount);

bool refineFtCSG(
    const std::array<vertex4d*, 5>& verts,
    const double threshold,
    bool& choice,
    bool& zeroX,
    std::array<double, timer_amount>& profileTimer,
    std::array<size_t, timer_amount>& profileCount,
    std::unordered_set<size_t>& domFuncIds);

bool calBezierCoordsAndDomFuncIds(
    const std::array<vertex4d*, 5>& verts,
    std::array<double, timer_amount>& profileTimer,
    std::array<size_t, timer_amount>& profileCount,
    Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor>& bezierCoords,
    Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor>& bezierFtVals,
    std::vector<size_t>& domFIds
);

bool refineFtCSG(
    const std::array<vertex4d*, 5>& verts,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor>> bezierCoords,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor>> bezierFtVals,
    const std::vector<size_t>& domFIds,
    const double threshold,
    bool& choice,
    bool& inside,
    bool& zeroX,
    std::array<double, timer_amount>& profileTimer,
    std::array<size_t, timer_amount>& profileCount,
    Eigen::Ref<Eigen::Matrix<int, 1, Eigen::Dynamic, Eigen::RowMajor>> domFuncFt0XIds,
    Eigen::Ref<Eigen::Matrix<int, 1, Eigen::Dynamic, Eigen::RowMajor>> func0XIds);


bool refineEqualSurfaceCSG(
    const std::array<vertex4d*, 5>& verts,
    const Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor>> bezierVals,
    const Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor>> bezierFtVals,
    const double threshold,
    const std::pair<size_t, size_t>& equalSurfFuncIds,
    bool& choice,
    bool& eqaulSurf0X,
    std::array<double, timer_amount>& profileTimer,
    std::array<size_t, timer_amount>& profileCount);


/// Refine critiera that only invokes the bezier computation. This is the samllest testing unit for comparing the run speed across different methods of constructing bezier simplex.
bool refineFtBezier(
    const std::array<vertex4d*, 5>& verts,
    const double threshold,
    bool& inside,
    bool& choice,
    bool& zeroX);

/// 3D (projected) distance check for the sweep function for 3D tetrahedra lives in the 4D domain, i.e., tets with 4D vertices. These tets are the results after the first stage subdivision.
/// @param[in] verts        The 4 tetrahedron vertices. Each `vertex4d` stores
///                         4D coordinates `coord` (x,y,z,t), a scalar value `valGradList.first` (= v),
///                         and its 4D gradient `valGradList.second` (= g). Only spatial parts of coords
///                         are used to form the tetra geometry.
/// @param[in] threshold            Threshold for the maxmimum allowed projected distance error for this tetrahedra.
///
/// @return         A boolean that decides if this tetrahedra needs to be subdivided.
bool refine3D(const std::array<vertex4d, 4>& verts, const double threshold);

bool refine3DCSG(const std::array<vertex4d*, 4>& verts, const double threshold, size_t csg_fn);
bool refine3DCSG(
    const std::array<vertex4d*, 4>& verts,
    const double threshold,
    const std::unordered_set<size_t>& domfIds);

bool refineCap(const std::array<vertex4d, 4> verts, const double threshold, bool& zeroX);
#endif /* ref_crit_h */


