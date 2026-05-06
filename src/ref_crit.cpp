//
//  ref_crit.cpp
//  adaptive_column_grid
//
//  Created by Yiwen Ju on 12/10/24.
//
// #define no_inside
#include "ref_crit.h"
#include "bezier_simplex.h"

// shared memory definition
// Eigen::Matrix<double, 2, 35> nPoints_eigen_shared;

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
    Eigen::Matrix<double, 2, 35> nPoints_eigen_shared;
    nPoints_eigen_shared << bezierVals, bezierGrad;
    zeroX = !outHullClip2D(nPoints_eigen_shared);
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
    // RefineTetGeoData& refineGeoData,
    const Eigen::Ref<const Eigen::RowVector<double, 35>> bezierVals,
    const Eigen::Ref<const Eigen::RowVector<double, 35>> bezierFtVals,
    bool& inside,
    std::array<double, timer_amount>& profileTimer,
    std::array<size_t, timer_amount>& profileCount)
{
    const auto& p1 = verts[0]->coord;
    const auto& p2 = verts[1]->coord;
    const auto& p3 = verts[2]->coord;
    const auto& p4 = verts[3]->coord;
    const auto& p5 = verts[4]->coord;

    if(isInside(bezierVals))
    {
        inside = true;
        return false;
    }

    if(bezierVals.maxCoeff() * bezierVals.minCoeff() > 0)
    {
        return false;
    }
    refineRes.fZeroX = true;
    // bezierFtValsShared
    // if (bezierDerOrds(bezierVals, vertsArray, bezierGrad_temp)) {
    //     // bezierGrad = bezierGrad_temp;
    //     return false;
    // }
    if(bezierFtVals.maxCoeff() * bezierFtVals.minCoeff() > 0)
    {
        return false;
    }
    Eigen::Matrix<double, 2, 35> nPoints_eigen_shared;
    nPoints_eigen_shared << bezierVals, bezierFtVals;
    refineRes.zeroX = !outHullClip2D(nPoints_eigen_shared);

    // need to optimize*, multiple recalculation 
    if (refineRes.zeroX) {
        // if(! refineGeoData.hasInit) refineGeoData.init();
        auto vec1 = p2 - p1, vec2 = p3 - p1, vec3 = p4 - p1, vec4 = p5 - p1;
        Eigen::Matrix4d vec;
        vec << vec1, vec2, vec3, vec4;
        Eigen::Matrix4d adj;
        adjugate(vec, adj);
        auto v1 = bezierFtVals[0];
        auto v2 = bezierFtVals[1];
        auto v3 = bezierFtVals[2];
        auto v4 = bezierFtVals[3];
        auto v5 = bezierFtVals[4];
        Eigen::RowVector4d gradList = Eigen::RowVector4d(v2 - v1, v3 - v1, v4 - v1, v5 - v1) * adj;

        Eigen::RowVector<double, 30> error =
            ((bezierFtVals.tail(30) - (bezierFtVals.head(5) * Bezier4D_ls.bottomRows(30).transpose()) / 3) *
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


// void RefineTetGeoData::init(const std::array<vertex4d*, 5>& verts)
// {
//     this->hasInit = true;
//     const auto& p1 = verts[0]->coord;
//     const auto& p2 = verts[1]->coord;
//     const auto& p3 = verts[2]->coord;
//     const auto& p4 = verts[3]->coord;
//     const auto& p5 = verts[4]->coord;
//     auto vec1 = p2 - p1, vec2 = p3 - p1, vec3 = p4 - p1, vec4 = p5 - p1;
//     Eigen::Matrix4d vec;
//     vec << vec1, vec2, vec3, vec4;
//     Eigen::Matrix4d adj;
//     adjugate(vec, adj);
//     this->determinant = vec.determinant();
//     Eigen::RowVector4d gradList = Eigen::RowVector4d(v2 - v1, v3 - v1, v4 - v1, v5 - v1) * adj;
//     this->gradNorm = gradList.norm();
// }



/// See header
bool calBezierCoordsAndDomFuncIds(
    const std::array<vertex4d*, 5>& verts,
    std::array<double, timer_amount>& profileTimer,
    std::array<size_t, timer_amount>& profileCount,
    Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor>& bezierCoords,
    Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor>& bezierFtVals,
    std::vector<size_t>& domFIds
)
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
    // Eigen::Index domf_num = static_cast<Eigen::Index>(v1s.size());
    // Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor> bezierCoords(domf_num, 35);
    // std::vector<size_t> domFIds; 
    getBezier4DDomFuncIds( p1,p2, p3, p4,p5,
    v1s, v2s, v3s, v4s, v5s,
    g1s, g2s, g3s, g4s, g5s, bezierCoords, domFIds);
    std::array<Eigen::RowVector4d, 5> tet_verts{p1, p2, p3, p4, p5};

    size_t n_rows = bezierCoords.rows();
    
    for(auto df_id : domFIds)
    {
        if(df_id < n_rows)
        {
            // bezierFtVals.row(df_id) = bezierCoords.row(df_id);
            bezierDerOrdsForFtVals(bezierCoords.row(df_id), tet_verts, bezierFtVals.row(df_id));
        } else {
            std::cout << " abnormal dom f id : " << df_id << std::endl;
        }
    }
    return true;
}


/// See header
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
    Eigen::Ref<Eigen::Matrix<int, 1, Eigen::Dynamic, Eigen::RowMajor>> func0XIds)
{
    Eigen::RowVectorXd ftErrors(domFIds.size());
    std::vector<RefineResInfo> refRes(domFIds.size());
    RefineTetGeoData refineGeoData;
    bool needRefine = false;
    inside = true;
    
    for(int i = 0; i < int(domFIds.size()); ++i)
    {
        size_t dFid = domFIds[i]; 

        // std::cout << " dfid " << dFid << std::endl;
        // RefineResInfo ref_res;
        bool inside_cur_domf = false;
        if(refineFtCSGSingle(verts, threshold, refRes[i],  
                    bezierCoords.row(dFid), bezierFtVals.row(dFid), inside_cur_domf, profileTimer, profileCount))
        {
            ftErrors[i] = refRes[i].error;
            needRefine = true;
        }
        inside = inside && inside_cur_domf;
        if(refRes[i].zeroX)
        {
            zeroX = refRes[i].zeroX;
            domFuncFt0XIds(dFid) = 1;
            // domFuncIds.insert(size_t(i));
        }
        func0XIds[dFid] = refRes[i].fZeroX;
    }

    if(needRefine)
    {
        Eigen::Index maxIndex = 0;
        ftErrors.maxCoeff(&maxIndex);
        choice = refRes[maxIndex].chooseTemporalRefine;
    }
    return needRefine;
}

bool refineEqualSurfaceCSG(
    const std::array<vertex4d*, 5>& verts,
    const Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor>> bezierVals,
    const Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor>> bezierFtVals,
    const double threshold,
    const std::pair<size_t, size_t>& equalSurfFuncIds,
    bool& choice,
    bool& eqaulSurf0X,
    double& max_error,
    std::array<double, timer_amount>& profileTimer,
    std::array<size_t, timer_amount>& profileCount)
{
    const auto& p1 = verts[0]->coord;
    const auto& p2 = verts[1]->coord;
    const auto& p3 = verts[2]->coord;
    const auto& p4 = verts[3]->coord;
    const auto& p5 = verts[4]->coord;
    eqaulSurf0X = false; 

    const auto& bezier_f1 = bezierVals.row(equalSurfFuncIds.first);
    // if(bezier_f1.maxCoeff() * bezier_f1.minCoeff() > 0) return false;
    const auto& bezier_f2 = bezierVals.row(equalSurfFuncIds.second);
    // if(bezier_f1.maxCoeff() * bezier_f1.minCoeff() > 0) return false;

    auto equalSurfBezier =  bezier_f1 - bezier_f2;
    if(equalSurfBezier.maxCoeff() * equalSurfBezier.minCoeff() > 0) return false;
    Eigen::Matrix<double, 2, 35> nPoints_eigen_shared;
    nPoints_eigen_shared << equalSurfBezier, bezier_f1;
    if(outHullClip2D(nPoints_eigen_shared)) return false;
    nPoints_eigen_shared << equalSurfBezier, bezier_f2;
    if(outHullClip2D(nPoints_eigen_shared)) return false;

    const auto& bezier_ft1 = bezierFtVals.row(equalSurfFuncIds.first);
    const auto& bezier_ft2 = bezierFtVals.row(equalSurfFuncIds.second);
    Eigen::RowVector<double, 35> ftSigns = bezier_ft1.cwiseProduct(bezier_ft2);
    if(ftSigns.minCoeff() > 0) return false;

    eqaulSurf0X = true;
    // if(! refineGeoData.hasInit) refineGeoData.init();
    auto vec1 = p2 - p1, vec2 = p3 - p1, vec3 = p4 - p1, vec4 = p5 - p1;
    Eigen::Matrix4d vec;
    vec << vec1, vec2, vec3, vec4;
    Eigen::Matrix4d adj;
    adjugate(vec, adj);
    auto v1 = equalSurfBezier[0];
    auto v2 = equalSurfBezier[1];
    auto v3 = equalSurfBezier[2];
    auto v4 = equalSurfBezier[3];
    auto v5 = equalSurfBezier[4];
    Eigen::RowVector4d gradList = Eigen::RowVector4d(v2 - v1, v3 - v1, v4 - v1, v5 - v1) * adj;

    Eigen::RowVector<double, 30> error =
        ((equalSurfBezier.tail(30) - (equalSurfBezier.head(5) * Bezier4D_ls.bottomRows(30).transpose()) / 3) *
            vec.determinant())
            .array()
            .abs();
    if (error.maxCoeff() > threshold * gradList.norm()) {
        Eigen::RowVector<double, 16> topFError = error(topFIndices);
        Eigen::RowVector<double, 16> botFError = error(botFIndices);
        auto cur_choice = std::max(error[3], error[16]) >
                    std::min(topFError.maxCoeff(), botFError.maxCoeff());
        // refineRes.error = error.maxCoeff(); 
        if(max_error <error.maxCoeff())
        {
            max_error = max_error > error.maxCoeff();
            choice = cur_choice;
        }
        return true;
    } 
    return false;
}


bool refineTripleSurfaceCSG(
    const std::array<vertex4d*, 5>& verts,
    const Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor>> bezierVals,
    const Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor>> bezierFtV,
    const double threshold,
    const std::array<size_t, 3>& tripleFuncIds,
    std::array<double, timer_amount>& profileTimer,
    std::array<size_t, timer_amount>& profileCount)
{
    const auto& p1 = verts[0]->coord;
    const auto& p2 = verts[1]->coord;
    const auto& p3 = verts[2]->coord;
    const auto& p4 = verts[3]->coord;
    const auto& p5 = verts[4]->coord;

    Eigen::MatrixXd JacobianMat(3, 4); 

    for(int i = 0; i < 3; ++i)
    {   
        Eigen::RowVector4d grad1 = verts[0]->grads.row(tripleFuncIds[i]);
        Eigen::RowVector4d grad2 = verts[1]->grads.row(tripleFuncIds[i]);
        Eigen::RowVector4d grad3 = verts[2]->grads.row(tripleFuncIds[i]);
        Eigen::RowVector4d grad4 = verts[3]->grads.row(tripleFuncIds[i]);
        Eigen::RowVector4d grad5 = verts[4]->grads.row(tripleFuncIds[i]);
        JacobianMat.row(i) = (grad1 + grad2 + grad3 + grad4 + grad5) / 5.0;
    }
    

    const auto& bezier_f1 = bezierVals.row(tripleFuncIds[0]);
    const auto& bezier_f2 = bezierVals.row(tripleFuncIds[1]);
    const auto& bezier_f3 = bezierVals.row(tripleFuncIds[2]);
   
    auto vec1 = p2 - p1, vec2 = p3 - p1, vec3 = p4 - p1, vec4 = p5 - p1;
    Eigen::Matrix4d vec;
    vec << vec1, vec2, vec3, vec4;
    Eigen::Matrix4d adj;
    adjugate(vec, adj);

    auto v1 = bezier_f1[0];
    auto v2 = bezier_f1[1];
    auto v3 = bezier_f1[2];
    auto v4 = bezier_f1[3];
    auto v5 = bezier_f1[4];
    Eigen::RowVector4d gradList1 = Eigen::RowVector4d(v2 - v1, v3 - v1, v4 - v1, v5 - v1) * adj;

    v1 = bezier_f2[0];
    v2 = bezier_f2[1];
    v3 = bezier_f2[2];
    v4 = bezier_f2[3];
    v5 = bezier_f2[4];
    Eigen::RowVector4d gradList2 = Eigen::RowVector4d(v2 - v1, v3 - v1, v4 - v1, v5 - v1) * adj;

    v1 = bezier_f3[0];
    v2 = bezier_f3[1];
    v3 = bezier_f3[2];
    v4 = bezier_f3[3];
    v5 = bezier_f3[4];
    Eigen::RowVector4d gradList3 = Eigen::RowVector4d(v2 - v1, v3 - v1, v4 - v1, v5 - v1) * adj;

    // Eigen::MatrixXd JacobianMat(3, 4); 
    // JacobianMat << gradList1, gradList2, gradList3;

    auto projecMat = JacobianMat.transpose() * (JacobianMat * JacobianMat.transpose()).inverse();


    Eigen::RowVector<double, 30> error1 =
        ((bezier_f1.tail(30) - (bezier_f1.head(5) * Bezier4D_ls.bottomRows(30).transpose()) / 3)
        *vec.determinant() / gradList1.norm())
            .array()
            .abs();

    Eigen::RowVector<double, 30> error2 =
        ((bezier_f2.tail(30) - (bezier_f2.head(5) * Bezier4D_ls.bottomRows(30).transpose()) / 3)
        *vec.determinant() / gradList2.norm())
            .array()
            .abs();

    Eigen::RowVector<double, 30> error3 =
        ((bezier_f3.tail(30) - (bezier_f3.head(5) * Bezier4D_ls.bottomRows(30).transpose()) / 3)
        *vec.determinant() / gradList3.norm() )
            .array()
            .abs();

    // auto bezier_f_all = bezier_f1 + bezier_f2 + bezier_f3;
    // Eigen::RowVector<double, 30> error_combine =
    //     ((bezier_f3.tail(30) - (bezier_f3.head(5) * Bezier4D_ls.bottomRows(30).transpose()) / 3)*vec.determinant())
    //         .array()
    //         .abs();

    // Eigen::MatrixXd errorMat(3, 30); 
    // errorMat << error1, error2, error3; 
    // auto distsMat = projecMat * errorMat;
    // Eigen::VectorXd error = distsMat.colwise().norm();

    auto error = error1 + error2 + error3;
    if (error.maxCoeff() > threshold ) {
        // Eigen::RowVector<double, 16> topFError = error(topFIndices);
        // Eigen::RowVector<double, 16> botFError = error(botFIndices);
        // auto cur_choice = std::max(error[3], error[16]) >
        //             std::min(topFError.maxCoeff(), botFError.maxCoeff());
        // refineRes.error = error.maxCoeff(); 
        // if(max_error <error.maxCoeff())
        // {
        //     max_error = max_error > error.maxCoeff();
        //     choice = cur_choice;
        // }
        return true;
    } 
    return false;
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

    Eigen::RowVector<double, 20> bezierVals3DShared;
    // Eigen::RowVector<double, 20> bezierVals;
    bezierVals3DShared.setZero();
    if (!bezier3D(p1, p2, p3, p4, v1, v2, v3, v4, g1, g2, g3, g4, bezierVals3DShared)) {
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
    double lhs = (bezierDiff(bezierVals3DShared) * D).array().abs().maxCoeff();
    double rhs = threshold * unNormF.norm();
    if (lhs > rhs) {
        return true;
    }
    return false;
}


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

    return {crossMatrix, edges.determinant() };
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
    Eigen::RowVector<double, 20> bezierVals3DShared;
    if (!bezier3D(p1, p2, p3, p4, v1, v2, v3, v4, g1, g2, g3, g4, bezierVals3DShared))
        return false;

    const Eigen::Vector3d unNormF = computeUnNormF(geom, v1, v2, v3, v4);

    const double lhs = (bezierDiff(bezierVals3DShared) * geom.determinant).array().abs().maxCoeff();
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