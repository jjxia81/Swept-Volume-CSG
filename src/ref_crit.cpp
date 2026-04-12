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
            refineRes.chooseTemporalRefine = std::max(error[3], error[16]) >
                     std::min(topFError.maxCoeff(), botFError.maxCoeff());
            refineRes.error = error.maxCoeff(); 
            return true;
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
    std::array<size_t, timer_amount>& profileCount,
    std::unordered_set<size_t>& domFuncIds)
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
        if(refineFtCSGSingle(verts, threshold, refRes[i], bezierCoords.row(dFid), profileTimer, profileCount))
        {
            ftErrors[i] = refRes[i].error;
            needRefine = true;
        }
        if(refRes[i].zeroX)
        {
            zeroX = refRes[i].zeroX;
            domFuncIds.insert(size_t(i));
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


// bool refine3DCSG(const std::array<vertex4d*, 4>& verts, const double threshold, size_t csg_fn)
// {
//     const auto& p1 = verts[0]->coord;
//     const auto& p2 = verts[1]->coord;
//     const auto& p3 = verts[2]->coord;
//     const auto& p4 = verts[3]->coord;

//     const auto& v1s = verts[0]->vals;
//     const auto& v2s = verts[1]->vals;
//     const auto& v3s = verts[2]->vals;
//     const auto& v4s = verts[3]->vals;

//     const auto& g1s = verts[0]->grads;
//     const auto& g2s = verts[1]->grads;
//     const auto& g3s = verts[2]->grads;
//     const auto& g4s = verts[3]->grads;

//     bool needRefine = false;
//     for(int fi = 0; fi < csg_fn; ++fi)
//     {
//         Eigen::RowVector<double, 20> bezierVals;
//         if (!bezier3D(p1, p2, p3, p4, v1s[fi], v2s[fi], v3s[fi], v4s[fi], g1s.row(fi), g2s.row(fi), g3s.row(fi), g4s.row(fi), bezierVals)) {
//             continue;
//         };
//         Eigen::Vector3d vec1 = p2.head(3) - p1.head(3), vec2 = p3.head(3) - p1.head(3),
//                         vec3 = p4.head(3) - p1.head(3);
//         Eigen::Matrix3d vec;
//         vec << vec1, vec2, vec3;
//         double D = vec.determinant();
//         Eigen::Matrix3d crossMatrix;
//         crossMatrix << vec2.cross(vec3), vec3.cross(vec1), vec1.cross(vec2);
//         Eigen::Vector3d unNormF =
//             Eigen::RowVector3d(v2s[fi] - v1s[fi], v3s[fi] - v1s[fi], v4s[fi] - v1s[fi]) * crossMatrix.transpose();
//         double lhs = (bezierDiff(bezierVals) * D).array().abs().maxCoeff();
//         double rhs = threshold * unNormF.norm();
//         if (lhs > rhs) {
//             needRefine = true;
//         }
//     }
//     return needRefine;
// }


// Computes edge vectors from the first vertex to the other three
Eigen::Matrix3d computeEdgeMatrix(
    const Eigen::Vector4d& p1,
    const Eigen::Vector4d& p2,
    const Eigen::Vector4d& p3,
    const Eigen::Vector4d& p4)
{
    Eigen::Matrix3d edges;
    edges.col(0) = p2.head(3) - p1.head(3);
    edges.col(1) = p3.head(3) - p1.head(3);
    edges.col(2) = p4.head(3) - p1.head(3);
    return edges;
}

// Computes the cross-product matrix and scalar determinant from edge vectors
struct TetrahedronGeometry {
    Eigen::Matrix3d crossMatrix;
    double determinant;
};

TetrahedronGeometry computeTetrahedronGeometry(const Eigen::Matrix3d& edges)
{
    const auto& vec1 = edges.col(0);
    const auto& vec2 = edges.col(1);
    const auto& vec3 = edges.col(2);

    Eigen::Matrix3d crossMatrix;
    crossMatrix.col(0) = vec2.cross(vec3);
    crossMatrix.col(1) = vec3.cross(vec1);
    crossMatrix.col(2) = vec1.cross(vec2);

    return { crossMatrix, edges.determinant() };
}

// Computes the unnormalized gradient direction for a single CSG function
Eigen::Vector3d computeUnNormF(
    const TetrahedronGeometry& geom,
    double v1, double v2, double v3, double v4)
{
    return geom.crossMatrix *
           Eigen::Vector3d(v2 - v1, v3 - v1, v4 - v1);
}

// Tests whether a single CSG function requires refinement
bool needsRefinementForFunction(
    const TetrahedronGeometry& geom,
    const Eigen::Vector4d& p1,
    const Eigen::Vector4d& p2,
    const Eigen::Vector4d& p3,
    const Eigen::Vector4d& p4,
    double v1, double v2, double v3, double v4,
    const Eigen::RowVectorXd& g1,
    const Eigen::RowVectorXd& g2,
    const Eigen::RowVectorXd& g3,
    const Eigen::RowVectorXd& g4,
    double threshold)
{
    Eigen::RowVector<double, 20> bezierVals;
    if (!bezier3D(p1, p2, p3, p4, v1, v2, v3, v4, g1, g2, g3, g4, bezierVals))
        return false;

    const Eigen::Vector3d unNormF = computeUnNormF(geom, v1, v2, v3, v4);

    const double lhs = (bezierDiff(bezierVals) * geom.determinant).array().abs().maxCoeff();
    const double rhs = threshold * unNormF.norm();

    return lhs > rhs;
}

bool refine3DCSG(
    const std::array<vertex4d*, 4>& verts,
    const double threshold,
    size_t domfId)
{
    const auto& p1 = verts[0]->coord;
    const auto& p2 = verts[1]->coord;
    const auto& p3 = verts[2]->coord;
    const auto& p4 = verts[3]->coord;

    // Precompute geometry once — shared across all CSG functions
    const Eigen::Matrix3d edges = computeEdgeMatrix(p1, p2, p3, p4);
    const TetrahedronGeometry geom = computeTetrahedronGeometry(edges);
    size_t fi = domfId;
    if (needsRefinementForFunction(
            geom,
            p1, p2, p3, p4,
            verts[0]->vals[fi], verts[1]->vals[fi],
            verts[2]->vals[fi], verts[3]->vals[fi],
            verts[0]->grads.row(fi), verts[1]->grads.row(fi),
            verts[2]->grads.row(fi), verts[3]->grads.row(fi),
            threshold))
    {
        return true; // Early exit: no need to check remaining functions
    }
    
    return false;
}


bool refine3DCSG(
    const std::array<vertex4d*, 4>& verts,
    const double threshold,
    const std::unordered_set<size_t>& domfIds)
{
    const auto& p1 = verts[0]->coord;
    const auto& p2 = verts[1]->coord;
    const auto& p3 = verts[2]->coord;
    const auto& p4 = verts[3]->coord;

    // Precompute geometry once — shared across all CSG functions
    const Eigen::Matrix3d edges = computeEdgeMatrix(p1, p2, p3, p4);
    const TetrahedronGeometry geom = computeTetrahedronGeometry(edges);
    
    for(auto fi : domfIds)
    {
        // size_t fi = domfIds[i];
        if (needsRefinementForFunction(
            geom,
            p1, p2, p3, p4,
            verts[0]->vals[fi], verts[1]->vals[fi],
            verts[2]->vals[fi], verts[3]->vals[fi],
            verts[0]->grads.row(fi), verts[1]->grads.row(fi),
            verts[2]->grads.row(fi), verts[3]->grads.row(fi),
            threshold))
        {
            return true; // Early exit: no need to check remaining functions
        }
    }
    return false;
}