//
//  ref_crit.cpp
//  adaptive_column_grid
//
//  Created by Yiwen Ju on 12/10/24.
//
// #define no_inside
#include "ref_crit.h"
#include "bezier_simplex.h"

bool refineFtBezier(
    const std::array<vertex4d*, 5>& verts,
    const double threshold,
    bool& inside,
    bool& choice,
    bool& zeroX)
{
    const auto& p1 = verts[0]->coord;
    const auto& p2 = verts[1]->coord;
    const auto& p3 = verts[2]->coord;
    const auto& p4 = verts[3]->coord;
    const auto& p5 = verts[4]->coord;

    const auto& v1 = verts[0]->valGradList.first;
    const auto& v2 = verts[1]->valGradList.first;
    const auto& v3 = verts[2]->valGradList.first;
    const auto& v4 = verts[3]->valGradList.first;
    const auto& v5 = verts[4]->valGradList.first;

    const auto& g1 = verts[0]->valGradList.second;
    const auto& g2 = verts[1]->valGradList.second;
    const auto& g3 = verts[2]->valGradList.second;
    const auto& g4 = verts[3]->valGradList.second;
    const auto& g5 = verts[4]->valGradList.second;
    Eigen::RowVector<double, 35> bezierVals;
    if (!bezier4D(p1, p2, p3, p4, p5, v1, v2, v3, v4, v5, g1, g2, g3, g4, g5, bezierVals, inside)) {
        return false;
    };
    return true;
}

/// See header
bool refineFt(
    const std::array<vertex4d*, 5>& verts,
    const double threshold,
    bool& inside,
    bool& choice,
    bool& zeroX,
    std::array<double, timer_amount>& profileTimer,
    std::array<size_t, timer_amount>& profileCount)
{
#if time_profile
    Timer first_bezier_timer(first_bezier, [&](auto timer, auto ms) {
        combine_timer(profileTimer, profileCount, timer, ms);
    });
#endif
    const auto& p1 = verts[0]->coord;
    const auto& p2 = verts[1]->coord;
    const auto& p3 = verts[2]->coord;
    const auto& p4 = verts[3]->coord;
    const auto& p5 = verts[4]->coord;

    const auto& v1 = verts[0]->valGradList.first;
    const auto& v2 = verts[1]->valGradList.first;
    const auto& v3 = verts[2]->valGradList.first;
    const auto& v4 = verts[3]->valGradList.first;
    const auto& v5 = verts[4]->valGradList.first;

    const auto& g1 = verts[0]->valGradList.second;
    const auto& g2 = verts[1]->valGradList.second;
    const auto& g3 = verts[2]->valGradList.second;
    const auto& g4 = verts[3]->valGradList.second;
    const auto& g5 = verts[4]->valGradList.second;
    Eigen::RowVector<double, 35> bezierVals;
    if (!bezier4D(p1, p2, p3, p4, p5, v1, v2, v3, v4, v5, g1, g2, g3, g4, g5, bezierVals, inside)) {
#if time_profile
        first_bezier_timer.Stop();
#endif
        return false;
    };
    Eigen::RowVector<double, 35> bezierGrad;
    if (bezierDerOrds(bezierVals, {p1, p2, p3, p4, p5}, bezierGrad)) {
#if time_profile
        first_bezier_timer.Stop();
#endif
        return false;
    }
    Eigen::Matrix<double, 2, 35> nPoints_eigen;
    nPoints_eigen << bezierVals, bezierGrad;
    zeroX = !outHullClip2D(nPoints_eigen);
#if time_profile
    first_bezier_timer.Stop();
    Timer first_func_timer(first_func, [&](auto timer, auto ms) {
        combine_timer(profileTimer, profileCount, timer, ms);
    });
#endif
    if (zeroX) {
        auto vec1 = p2 - p1, vec2 = p3 - p1, vec3 = p4 - p1, vec4 = p5 - p1;
        Eigen::Matrix4d vec;
        vec << vec1, vec2, vec3, vec4;
        Eigen::Matrix4d adj;
        adjugate(vec, adj);
        auto v1 = bezierGrad[0];
        auto v2 = bezierGrad[1];
        auto v3 = bezierGrad[2];
        auto v4 = bezierGrad[3];
        auto v5 = bezierGrad[4];
        Eigen::RowVector4d gradList = Eigen::RowVector4d(v2 - v1, v3 - v1, v4 - v1, v5 - v1) * adj;
        Eigen::RowVector<double, 30> error =
            ((bezierGrad.tail(30) - (bezierGrad.head(5) * Bezier4D_ls.bottomRows(30).transpose()) / 3) *
             vec.determinant())
                .array()
                .abs();
        if (error.maxCoeff() > threshold * gradList.norm()) {
            Eigen::RowVector<double, 16> topFError = error(topFIndices);
            Eigen::RowVector<double, 16> botFError = error(botFIndices);
            choice = std::max(error[3], error[16]) >
                     std::min(topFError.maxCoeff(), botFError.maxCoeff());
#if time_profile
            first_func_timer.Stop();
#endif
            return true;
        }
    }
#if time_profile
    first_func_timer.Stop();
#endif
    return false;
}

/// See header
bool refineFtCSGSingle(
    const std::array<vertex4d*, 5>& verts,
    const double threshold,
    RefineResInfo& refineRes,
    const Eigen::Ref<Eigen::RowVector<double, 35>> bezierVals,
    std::array<double, timer_amount>& profileTimer,
    std::array<size_t, timer_amount>& profileCount)
{

    const auto& p1 = verts[0]->coord;
    const auto& p2 = verts[1]->coord;
    const auto& p3 = verts[2]->coord;
    const auto& p4 = verts[3]->coord;
    const auto& p5 = verts[4]->coord;

    if(bezierVals.maxCoeff() * bezierVals.minCoeff() > 0)
    {
        return false;
    }
    refineRes.fZeroX = true;

    Eigen::RowVector<double, 35> bezierGrad;
    if (bezierDerOrds(bezierVals, {p1, p2, p3, p4, p5}, bezierGrad)) {

        return false;
    }
    Eigen::Matrix<double, 2, 35> nPoints_eigen;
    nPoints_eigen << bezierVals, bezierGrad;
    refineRes.zeroX = !outHullClip2D(nPoints_eigen);

    if (refineRes.zeroX) {
        auto vec1 = p2 - p1, vec2 = p3 - p1, vec3 = p4 - p1, vec4 = p5 - p1;
        Eigen::Matrix4d vec;
        vec << vec1, vec2, vec3, vec4;
        Eigen::Matrix4d adj;
        adjugate(vec, adj);
        auto v1 = bezierGrad[0];
        auto v2 = bezierGrad[1];
        auto v3 = bezierGrad[2];
        auto v4 = bezierGrad[3];
        auto v5 = bezierGrad[4];
        Eigen::RowVector4d gradList = Eigen::RowVector4d(v2 - v1, v3 - v1, v4 - v1, v5 - v1) * adj;
        Eigen::RowVector<double, 30> error =
            ((bezierGrad.tail(30) - (bezierGrad.head(5) * Bezier4D_ls.bottomRows(30).transpose()) / 3) *
             vec.determinant())
                .array()
                .abs();
        if (error.maxCoeff() > threshold * gradList.norm()) {
            Eigen::RowVector<double, 16> topFError = error(topFIndices);
            Eigen::RowVector<double, 16> botFError = error(botFIndices);
            refineRes.choice = std::max(error[3], error[16]) >
                     std::min(topFError.maxCoeff(), botFError.maxCoeff());
            refineRes.error = error.maxCoeff(); 
            return true;
        } else {
            refineRes.meetThreshold = true;
        }
    }
    return false;
}


/// See header
bool refineFtCSG(
    const std::array<vertex4d*, 5>& verts,
    const double threshold,
    bool& choice,
    bool& zeroX,
    std::array<double, timer_amount>& profileTimer,
    std::array<size_t, timer_amount>& profileCount)
{
    const auto& p1 = verts[0]->coord;
    const auto& p2 = verts[1]->coord;
    const auto& p3 = verts[2]->coord;
    const auto& p4 = verts[3]->coord;
    const auto& p5 = verts[4]->coord;

    const auto& v1s = verts[0]->vals;
    const auto& v2s = verts[1]->vals;
    const auto& v3s = verts[2]->vals;
    const auto& v4s = verts[3]->vals;
    const auto& v5s = verts[4]->vals;

    const auto& g1s = verts[0]->grads;
    const auto& g2s = verts[1]->grads;
    const auto& g3s = verts[2]->grads;
    const auto& g4s = verts[3]->grads;
    const auto& g5s = verts[4]->grads;
    Eigen::Index domf_num = static_cast<Eigen::Index>(v1s.size());
    Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor> bezierCoords(domf_num, 35);
    std::vector<size_t> domFIds; 
    getBezier4DDomFuncIds( p1,p2, p3, p4,p5,
    v1s, v2s, v3s, v4s, v5s,
    g1s, g2s, g3s, g4s, g5s, bezierCoords, domFIds);
    
    Eigen::RowVectorXd ftErrors(domFIds.size());
    std::vector<RefineResInfo> refRes(domFIds.size());
    bool needRefine = false;
    for(int i = 0; i < int(domFIds.size()); ++i)
    {
        size_t dFid = domFIds[i]; 
        // RefineResInfo ref_res;
        if(refineFtCSGSingle(verts, threshold, refRes[i], bezierVals.row(dFid), profileTimer, profileCount))
        {
            ftErrors[i] = refRes[i].error;
            needRefine = true;
        }
    }
    if(needRefine)
    {
        Eigen::Index maxIndex = 0;
        ftErrors.maxCoeff(&maxIndex);
        choice = refRes[maxIndex].chooseTemporalRefine;
    }
    return needRefine;
}

/// Construct the values of one function at the bezier control points within a tet.
///
/// @param[in] p1-p4          The vertex cooridantes of four tet vertices.
/// @param[in] v1-v4         The function values at four tet vertices.
/// @param[in] g1-g4            The total derivative of the functions in x, y, z, t direction at four tet vertices.
/// @param[out] bezier          The eigen vector of 20 bezier values.
///
/// @return         If 20 bezier values contain zero-crossing.
bool bezier3D(
    const Eigen::RowVector4d& p1,
    const Eigen::RowVector4d& p2,
    const Eigen::RowVector4d& p3,
    const Eigen::RowVector4d& p4,
    const double v1,
    const double v2,
    const double v3,
    const double v4,
    const Eigen::RowVector4d& g1,
    const Eigen::RowVector4d& g2,
    const Eigen::RowVector4d& g3,
    const Eigen::RowVector4d& g4,
    Eigen::RowVector<double, 20>& bezier)
{
    // Compute edge values
    std::array<double, 4> v1s = {
        v1 + g1.dot(p2 - p1) / 3,
        v1 + g1.dot(p3 - p1) / 3,
        v1 + g1.dot(p4 - p1) / 3};

    std::array<double, 4> v2s = {
        v2 + g2.dot(p3 - p2) / 3,
        v2 + g2.dot(p4 - p2) / 3,
        v2 + g2.dot(p1 - p2) / 3};

    std::array<double, 4> v3s = {
        v3 + g3.dot(p4 - p3) / 3,
        v3 + g3.dot(p1 - p3) / 3,
        v3 + g3.dot(p2 - p3) / 3};

    std::array<double, 4> v4s = {
        v4 + g4.dot(p1 - p4) / 3,
        v4 + g4.dot(p2 - p4) / 3,
        v4 + g4.dot(p3 - p4) / 3};
    // Compute face values
    double face1 =
        (9 * (v2s[0] + v2s[1] + v3s[0] + v3s[2] + v4s[1] + v4s[2]) / 6 - v2 - v3 - v4) / 6;
    double face2 =
        (9 * (v1s[1] + v1s[2] + v3s[0] + v3s[1] + v4s[0] + v4s[2]) / 6 - v1 - v3 - v4) / 6;
    double face3 =
        (9 * (v1s[0] + v1s[2] + v2s[1] + v2s[2] + v4s[0] + v4s[1]) / 6 - v1 - v2 - v4) / 6;
    double face4 =
        (9 * (v1s[0] + v1s[1] + v2s[0] + v2s[2] + v3s[1] + v3s[2]) / 6 - v1 - v2 - v3) / 6;

    // Combine results into a single row vector
    bezier << v1, v2, v3, v4, v1s[0], v1s[1], v1s[2], v2s[0], v2s[1], v2s[2], v3s[0], v3s[1],
        v3s[2], v4s[0], v4s[1], v4s[2], face1, face2, face3, face4;

    if (get_sign(bezier.maxCoeff()) == get_sign(bezier.minCoeff())) {
        return false;
    }
    return true;
}

/// Construct the value differences between linear interpolations and bezier approximations at 16 bezier control points (excluding control points at tet vertices)
/// @param[in] valList          The eigen vector of 20 bezier values.
///
/// @return         The value differences at 16 control points.
Eigen::Vector<double, 16> bezierDiff(const Eigen::Vector<double, 20> valList)
{
    /// Constant coefficient to obtain linear interpolated values at each bezier control points
    const Eigen::Matrix<double, 16, 4> linear_coeff{
        {2, 1, 0, 0},
        {2, 0, 1, 0},
        {2, 0, 0, 1},
        {0, 2, 1, 0},
        {0, 2, 0, 1},
        {1, 2, 0, 0},
        {0, 0, 2, 1},
        {1, 0, 2, 0},
        {0, 1, 2, 0},
        {1, 0, 0, 2},
        {0, 1, 0, 2},
        {0, 0, 1, 2},
        {0, 1, 1, 1},
        {1, 0, 1, 1},
        {1, 1, 0, 1},
        {1, 1, 1, 0}};
    Eigen::Vector<double, 16> linear_val = (linear_coeff * valList.head(4)) / 3;
    return valList.tail(16) - linear_val;
}

/// See header
bool refine3D(const std::array<vertex4d, 4>& verts, const double threshold)
{
    const auto& p1 = verts[0].coord;
    const auto& p2 = verts[1].coord;
    const auto& p3 = verts[2].coord;
    const auto& p4 = verts[3].coord;

    const auto& v1 = verts[0].valGradList.first;
    const auto& v2 = verts[1].valGradList.first;
    const auto& v3 = verts[2].valGradList.first;
    const auto& v4 = verts[3].valGradList.first;

    const auto& g1 = verts[0].valGradList.second;
    const auto& g2 = verts[1].valGradList.second;
    const auto& g3 = verts[2].valGradList.second;
    const auto& g4 = verts[3].valGradList.second;

    Eigen::RowVector<double, 20> bezierVals;
    if (!bezier3D(p1, p2, p3, p4, v1, v2, v3, v4, g1, g2, g3, g4, bezierVals)) {
        return false;
    };
    Eigen::Vector3d vec1 = p2.head(3) - p1.head(3), vec2 = p3.head(3) - p1.head(3),
                    vec3 = p4.head(3) - p1.head(3);
    Eigen::Matrix3d vec;
    vec << vec1, vec2, vec3;
    double D = vec.determinant();
    Eigen::Matrix3d crossMatrix;
    crossMatrix << vec2.cross(vec3), vec3.cross(vec1), vec1.cross(vec2);
    Eigen::Vector3d unNormF =
        Eigen::RowVector3d(v2 - v1, v3 - v1, v4 - v1) * crossMatrix.transpose();
    double lhs = (bezierDiff(bezierVals) * D).array().abs().maxCoeff();
    double rhs = threshold * unNormF.norm();
    if (lhs > rhs) {
        return true;
    }
    return false;
}
