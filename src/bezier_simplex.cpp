#include "bezier_simplex.h"








/// Compute the adjugate (classical adjoint) of a 4×4 matrix using explicit minor and cofactor expansion.
/// The adjugate matrix is the transpose of the cofactor matrix and is used in computing the matrix inverse
/// or determinant-related geometric transformations in 4D computations. This implementation iteratively
/// evaluates all 3×3 minors of the input matrix and constructs the transposed cofactor matrix explicitly.
///
/// @param[in] mat          The input 4×4 matrix. Typically represents a Jacobian or transformation matrix
///                         in 4D space (e.g., spatial-temporal derivatives or local frame transformations).
/// @param[out] adjugate    The resulting 4×4 adjugate (classical adjoint) matrix. Each element (j, i)
///                         corresponds to the signed determinant of the 3×3 minor obtained by removing
///                         the i-th row and j-th column from the input matrix.
void adjugate(const Eigen::Matrix<double, 4, 4>& mat, Eigen::Matrix4d& adjugate)
{
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            Eigen::Matrix3d minor;

            int rowOffset = 0;
            for (int row = 0; row < 4; ++row) {
                if (row == i) {
                    rowOffset = 1;
                    continue;
                }
                int colOffset = 0;
                for (int col = 0; col < 4; ++col) {
                    if (col == j) {
                        colOffset = 1;
                        continue;
                    }
                    minor(row - rowOffset, col - colOffset) = mat(row, col);
                }
            }
            // Compute the cofactor
            double cofactor = std::pow(-1, i + j) * minor.determinant();
            adjugate(j, i) = cofactor; // Note the transpose here
        }
    }
}

/// Perform degree elevation from a quadratic Bezier simplex to a cubic Bezier simplex in 4D space.
/// This function computes the elevated Bezier control ordinates by linearly combining the 16 original
/// cubic coefficients using a precomputed elevation matrix (`elevMatrix`) and weighting matrix (`ls`).
/// The resulting 35 cubic control ordinates preserve the original geometric form while improving
/// smoothness and compatibility with higher-order Bezier simplex evaluations.
///
/// @param[in] ords         The 16 Bezier control ordinates (coefficients) corresponding to the cubic Bezier tetrahedron.
///                         Each ordinate represents the scalar value of the Bezier function at a vertex in barycentric order.
/// @param[out] bezier      The 35 elevated Bezier control ordinates for the cubic Bezier simplex.
///                         These ordinates are computed by applying the linear transformation defined by `elevMatrix`
///                         and `ls`, followed by normalization.
void bezierElev(const Eigen::RowVector<double, 16>& ords, Eigen::RowVector<double, 35>& bezier)
{
    Eigen::Vector<double, 5> elevRow;
    for (size_t i = 0; i < 35; i++) {
        elevRow << ords[elevMatrix[i][0]], ords[elevMatrix[i][1]], ords[elevMatrix[i][2]],
            ords[elevMatrix[i][3]], ords[elevMatrix[i][4]];
        bezier[i] = Bezier4D_ls.row(i).dot(elevRow);
    }
    bezier = bezier / 3.0;
}

/// Compute the elevated Bezier control ordinates for the directional derivative of a cubic (order-3)
/// 4D Bezier simplex along its temporal edge. The derivative yields a quadratic (order-2) Bezier simplex,
/// which is subsequently degree-elevated to cubic to produce 35 ordinates for uniform processing.
///
/// @param[in] ords         35 cubic Bezier ordinates of the scalar field on the 4D 5-cell (barycentric order).
/// @param[in] verts        The 5 vertices of the simplex (RowVector4d), used to normalize by ‖verts[0]−verts[4]‖.
/// @param[out] bezierGrad  35 cubic Bezier ordinates of the directional derivative after degree elevation.
///
/// @details                Differences indexed by `derMatrix` form 16 cubic derivative ordinates (scaled by 3/‖Δt‖),
///                         then `bezierElev` elevates quadratic→cubic. Sign consistency of `bezierGrad` is returned.
///
/// @return                 true if max and min coefficients of `bezierGrad` share the same sign (no zero crossing);
///                         false otherwise.
bool bezierDerOrds(
    const Eigen::RowVector<double, 35>& ords,
    const std::array<Eigen::RowVector4d, 5>& verts,
    Eigen::RowVector<double, 35>& bezierGrad)
{
    Eigen::RowVector<double, 16> vals;
    double norm = (verts[0] - verts[4]).norm();
    // Loop through rows of derMatrix
    for (int i = 0; i < 15; i++) {
        vals[i] = (ords[derMatrix[i][0]] - ords[derMatrix[i][1]]) * 3.0 /
                  norm; // Normalize by the norm of verts
    }
    vals[15] = 0;
    bezierElev(vals, bezierGrad);
    return get_sign(bezierGrad.maxCoeff()) == get_sign(bezierGrad.minCoeff());
}

/// Extremely fast outside-hull clipping test for a 2D set of Bezier ordinates projected to the plane.
/// Given 35 2D points (e.g., a cubic Bezier simplex control points in 4D),
/// this routine checks whether all points lie within some origin-centered half-plane
/// (i.e., there exists a direction u such that u·p >= 0 for all points p).
/// Equivalently, it verifies that the convex hull of the points does **not** wrap around
/// the origin (no full 360° coverage), so a line through the origin separates the origin
/// from the point set.
///
/// The algorithm maintains a feasible angular wedge of outward normals, initialized by the
/// first point’s perpendiculars, and iteratively clips this wedge with each subsequent point.
/// If the wedge becomes empty, the origin is enclosed by the hull and the test fails.
///
/// @param[in] pts          A 2×35 matrix whose columns are the 2D points to test.
///                         Columns are 35 bezier control points.
/// @return                 true  if a separating origin-centered half-plane exists (origin is outside the hull);
///                         false if the feasible wedge collapses (origin is enclosed/covered by the hull).
///
/// @note                   Robustness control: uses a small epsilon (1e-7) to handle near-collinear cases and ties by
///                         falling back to an orientation test via a perpendicular vector.
///                         The helper `perp(v) = (-v_y, v_x)` rotates vectors by +90°.
bool outHullClip2D(Eigen::Matrix<double, 2, 35> pts)
{
    constexpr double eps = 0.0000001;
    bool r1, r2;
    double t;
    auto perp = [](const Eigen::Vector2d& data) { return Eigen::Vector2d{-data[1], data[0]}; };
    std::array<Eigen::Vector2d, 2> range = {-perp(pts.col(0)), perp(pts.col(0))};
    for (size_t i = 1; i < 35; i++) {
        t = range[0].dot(pts.col(i));
        if (t > eps) {
            r1 = true;
        } else if (t < -eps) {
            r1 = false;
        } else if (perp(range[0]).dot(pts.col(i)) > 0) {
            r1 = true;
        } else {
            return false;
        }
        t = range[1].dot(pts.col(i));
        if (t > eps) {
            r2 = true;
        } else if (t < -eps) {
            r2 = false;
        } else if (perp(range[1]).dot(pts.col(i)) < 0) {
            r2 = true;
        } else {
            return false;
        }

        if (!r1 && !r2) {
            return false;
        } else if (!r1) {
            range[0] = -perp(pts.col(i));
        } else if (!r2) {
            range[1] = perp(pts.col(i));
        }
    }
    return true;
}

void getBezier4DDomFuncIds(
    const Eigen::RowVector4d& p1,
    const Eigen::RowVector4d& p2,
    const Eigen::RowVector4d& p3,
    const Eigen::RowVector4d& p4,
    const Eigen::RowVector4d& p5,
    const Eigen::RowVectorXd& v1s,
    const Eigen::RowVectorXd& v2s,
    const Eigen::RowVectorXd& v3s,
    const Eigen::RowVectorXd& v4s,
    const Eigen::RowVectorXd& v5s,
    const MatrixX4dRowMajor& g1s,
    const MatrixX4dRowMajor& g2s,
    const MatrixX4dRowMajor& g3s,
    const MatrixX4dRowMajor& g4s,
    const MatrixX4dRowMajor& g5s,
    Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor>& bezierCoords,
    std::vector<size_t>& domFIds)
{
    Eigen::Index domf_num = static_cast<Eigen::Index>(v1s.size());
    // Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor> bezierCoords(domf_num, 35);
    for(size_t fid = 0; fid < domf_num; ++fid)
    {
        bezier4D(p1, p2, p3, p4, p5, v1s[fid], v2s[fid], v3s[fid], v4s[fid], v5s[fid], 
            g1s.row(fid), g2s.row(fid), g3s.row(fid), g4s.row(fid), g5s.row(fid), bezierCoords.row(fid));
    }
    std::unordered_set<size_t> dFIds; 
    for(int i = 0; i < 35; ++i)
    {
        RVecX data = bezierCoords.col(i);
        auto [val, id] = csg_fun(data);
        dFIds.insert(id);
    }
    domFIds.assign(dFIds.begin(), dFIds.end());
}