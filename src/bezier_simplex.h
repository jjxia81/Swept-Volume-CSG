#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unordered_set>
using Scalar = double;
using RVecX = Eigen::RowVectorXd;

inline std::pair<Scalar, size_t> csg_fun_max(const RVecX& input)
{
    Eigen::Index fid = 0;
    Scalar val = input.maxCoeff(&fid);
    return {val, static_cast<size_t>(fid)};
}

inline std::pair<Scalar, size_t> csg_fun_min(const RVecX& input)
{
    
    Eigen::Index fid = 0; 
    Scalar val = input.minCoeff(&fid);
    return {val, static_cast<size_t>(fid)};
}

using CSGFunType = std::function<std::pair<Scalar, size_t>(const RVecX&)>;
inline CSGFunType csg_fun = csg_fun_max;

void set_csg_val_func(CSGFunType csg_f);


/// bezier ordinates of a 4D cubic simplex
const Eigen::Matrix<double, 35, 5, Eigen::RowMajor> Bezier4D_ls{/// Vertices
                                                       {3, 0, 0, 0, 0},
                                                       {0, 3, 0, 0, 0},
                                                       {0, 0, 3, 0, 0},
                                                       {0, 0, 0, 3, 0},
                                                       {0, 0, 0, 0, 3},

                                                       /// Edges
                                                       {2, 1, 0, 0, 0},
                                                       {2, 0, 1, 0, 0},
                                                       {2, 0, 0, 1, 0},
                                                       {2, 0, 0, 0, 1},
                                                       {1, 2, 0, 0, 0},
                                                       {0, 2, 1, 0, 0},
                                                       {0, 2, 0, 1, 0},
                                                       {0, 2, 0, 0, 1},
                                                       {1, 0, 2, 0, 0},
                                                       {0, 1, 2, 0, 0},
                                                       {0, 0, 2, 1, 0},
                                                       {0, 0, 2, 0, 1},
                                                       {1, 0, 0, 2, 0},
                                                       {0, 1, 0, 2, 0},
                                                       {0, 0, 1, 2, 0},
                                                       {0, 0, 0, 2, 1},
                                                       {1, 0, 0, 0, 2},
                                                       {0, 1, 0, 0, 2},
                                                       {0, 0, 1, 0, 2},
                                                       {0, 0, 0, 1, 2},

                                                       /// Faces
                                                       {1, 1, 1, 0, 0},
                                                       {1, 1, 0, 1, 0},
                                                       {1, 1, 0, 0, 1},
                                                       {1, 0, 1, 1, 0},
                                                       {1, 0, 1, 0, 1},
                                                       {1, 0, 0, 1, 1},
                                                       {0, 1, 1, 1, 0},
                                                       {0, 1, 1, 0, 1},
                                                       {0, 1, 0, 1, 1},
                                                       {0, 0, 1, 1, 1}};



/// Hard coded parameters that take the derivative of a bezier simplex.
const std::array<std::array<size_t, 2>, 15> derMatrix{
    {{0, 8},
     {5, 27},
     {6, 29},
     {7, 30},
     {8, 21},
     {9, 12},
     {25, 32},
     {26, 33},
     {27, 22},
     {13, 16},
     {28, 34},
     {29, 23},
     {17, 20},
     {30, 24},
     {21, 4}}};

/// Hard coded parameters that elevate the simplex to cubic.
const std::array<std::array<size_t, 5>, 35> elevMatrix{
    {{0, 15, 15, 15, 15},  {15, 5, 15, 15, 15}, {15, 15, 9, 15, 15}, {15, 15, 15, 12, 15},
     {15, 15, 15, 15, 14}, {1, 0, 15, 15, 15},  {2, 15, 0, 15, 15},  {3, 15, 15, 0, 15},
     {4, 15, 15, 15, 0},   {5, 1, 15, 15, 15},  {15, 6, 5, 15, 15},  {15, 7, 15, 5, 15},
     {15, 8, 15, 15, 5},   {9, 15, 2, 15, 15},  {15, 9, 6, 15, 15},  {15, 15, 10, 9, 15},
     {15, 15, 11, 15, 9},  {12, 15, 15, 3, 15}, {15, 12, 15, 7, 15}, {15, 15, 12, 10, 15},
     {15, 15, 15, 13, 12}, {14, 15, 15, 15, 4}, {15, 14, 15, 15, 8}, {15, 15, 14, 15, 11},
     {15, 15, 15, 14, 13}, {6, 2, 1, 15, 15},   {7, 3, 15, 1, 15},   {8, 4, 15, 15, 1},
     {10, 15, 3, 2, 15},   {11, 15, 4, 15, 2},  {13, 15, 15, 4, 3},  {15, 10, 7, 6, 15},
     {15, 11, 8, 15, 6},   {15, 13, 15, 8, 7},  {15, 15, 13, 11, 10}}};

/// Hard coded indices of top and bottom tetrahedra of the 4D Bezier simplex control coordinates(35 indices in total).
const std::array<int, 16> topFIndices = {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 20, 21, 23, 26};
const std::array<int, 16> botFIndices =
    {5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 26, 27, 28, 29};

/// returns a `bool` value that `true` represents positive and `false` represents negative of the input value `x`.
inline bool get_sign(const double x)
{
    return x > 0;
}

void adjugate(const Eigen::Matrix<double, 4, 4>& mat, Eigen::Matrix4d& adjugate);
void bezierElev(const Eigen::RowVector<double, 16>& ords, Eigen::Ref<Eigen::RowVector<double, 35>> bezier);

bool bezierDerOrds(
const Eigen::RowVector<double, 35>& ords,
const std::array<Eigen::RowVector4d, 5>& verts,
Eigen::Ref<Eigen::RowVector<double, 35>> bezierGrad);

bool outHullClip2D(const Eigen::Matrix<double, 2, 35>& pts);


/// Construct the values of one function at the bezier control points within a tet.
///
/// @param[in] p1-p4          The vertex cooridantes of four tet vertices.
/// @param[in] v1-v4         The function values at four tet vertices.
/// @param[in] g1-g4            The total derivative of the functions in x, y, z, t direction at four tet vertices.
/// @param[out] bezier          The eigen vector of 20 bezier values.
///
/// @return         If 20 bezier values contain zero-crossing.
inline bool bezier3D(
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
inline Eigen::Vector<double, 16> bezierDiff(const Eigen::Vector<double, 20>& valList)
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



[[gnu::always_inline]]
inline void bezier4D(
    const Eigen::RowVector4d& p1,
    const Eigen::RowVector4d& p2,
    const Eigen::RowVector4d& p3,
    const Eigen::RowVector4d& p4,
    const Eigen::RowVector4d& p5,
    const double v1,
    const double v2,
    const double v3,
    const double v4,
    const double v5,
    const Eigen::RowVector4d& g1,
    const Eigen::RowVector4d& g2,
    const Eigen::RowVector4d& g3,
    const Eigen::RowVector4d& g4,
    const Eigen::RowVector4d& g5,
    // Eigen::RowVector<double, 35>& bezier
    Eigen::Ref<Eigen::RowVector<double, 35>> bezier)
{
    // ───── compile–time reciprocals (mul beats div) ───────────────────────────
    constexpr double inv3 = 1.0 / 3.0;
    constexpr double inv6 = 1.0 / 6.0;
    constexpr double half = 0.5;

    // ───── pairwise differences (one evaluation each) ─────────────────────────
    const Eigen::RowVector4d p12 = p1 - p2, p13 = p1 - p3, p14 = p1 - p4, p15 = p1 - p5,
                             p23 = p2 - p3, p24 = p2 - p4, p25 = p2 - p5, p34 = p3 - p4,
                             p35 = p3 - p5, p45 = p4 - p5;

    // ───── per-vertex edge control points (dot once, inv3 mul) ────────────────
    const double v1s0 = v1 - inv3 * g1.dot(p12);
    const double v1s1 = v1 - inv3 * g1.dot(p13);
    const double v1s2 = v1 - inv3 * g1.dot(p14);
    const double v1s3 = v1 - inv3 * g1.dot(p15);

    const double v2s0 = v2 + inv3 * g2.dot(p12);
    const double v2s1 = v2 - inv3 * g2.dot(p23);
    const double v2s2 = v2 - inv3 * g2.dot(p24);
    const double v2s3 = v2 - inv3 * g2.dot(p25);

    const double v3s0 = v3 + inv3 * g3.dot(p13);
    const double v3s1 = v3 + inv3 * g3.dot(p23);
    const double v3s2 = v3 - inv3 * g3.dot(p34);
    const double v3s3 = v3 - inv3 * g3.dot(p35);

    const double v4s0 = v4 + inv3 * g4.dot(p14);
    const double v4s1 = v4 + inv3 * g4.dot(p24);
    const double v4s2 = v4 + inv3 * g4.dot(p34);
    const double v4s3 = v4 - inv3 * g4.dot(p45);

    const double v5s0 = v5 + inv3 * g5.dot(p15);
    const double v5s1 = v5 + inv3 * g5.dot(p25);
    const double v5s2 = v5 + inv3 * g5.dot(p35);
    const double v5s3 = v5 + inv3 * g5.dot(p45);

    // helper macro for tiny triple sums
#define SUM3(a, b, c) ((a) + (b) + (c))

    // ───── face control points (reuse edge sums, no extra temporaries) ────────
    const double e1 = (v1s0 + v1s1 + v2s0 + v2s1 + v3s0 + v3s1) * inv6;
    const double face1 = e1 + (e1 - SUM3(v1, v2, v3) * inv3) * half;

    const double e2 = (v1s0 + v1s2 + v2s0 + v2s2 + v4s0 + v4s1) * inv6;
    const double face2 = e2 + (e2 - SUM3(v1, v2, v4) * inv3) * half;

    const double e3 = (v1s0 + v1s3 + v2s0 + v2s3 + v5s0 + v5s1) * inv6;
    const double face3 = e3 + (e3 - SUM3(v1, v2, v5) * inv3) * half;

    const double e4 = (v1s1 + v1s2 + v3s0 + v3s2 + v4s0 + v4s2) * inv6;
    const double face4 = e4 + (e4 - SUM3(v1, v3, v4) * inv3) * half;

    const double e5 = (v1s1 + v1s3 + v3s0 + v3s3 + v5s0 + v5s2) * inv6;
    const double face5 = e5 + (e5 - SUM3(v1, v3, v5) * inv3) * half;

    const double e6 = (v1s2 + v1s3 + v4s0 + v4s3 + v5s0 + v5s3) * inv6;
    const double face6 = e6 + (e6 - SUM3(v1, v4, v5) * inv3) * half;

    const double e7 = (v2s1 + v2s2 + v3s1 + v3s2 + v4s1 + v4s2) * inv6;
    const double face7 = e7 + (e7 - SUM3(v2, v3, v4) * inv3) * half;

    const double e8 = (v2s1 + v2s3 + v3s1 + v3s3 + v5s1 + v5s2) * inv6;
    const double face8 = e8 + (e8 - SUM3(v2, v3, v5) * inv3) * half;

    const double e9 = (v2s2 + v2s3 + v4s1 + v4s3 + v5s1 + v5s3) * inv6;
    const double face9 = e9 + (e9 - SUM3(v2, v4, v5) * inv3) * half;

    const double e10 = (v3s2 + v3s3 + v4s2 + v4s3 + v5s2 + v5s3) * inv6;
    const double face10 = e10 + (e10 - SUM3(v3, v4, v5) * inv3) * half;

#undef SUM3

    // ───── populate the 35-coeff vector (single << stream) ────────────────────
    bezier << v1, v2, v3, v4, v5, v1s0, v1s1, v1s2, v1s3, v2s0, v2s1, v2s2, v2s3, v3s0, v3s1, v3s2,
        v3s3, v4s0, v4s1, v4s2, v4s3, v5s0, v5s1, v5s2, v5s3, face1, face2, face3, face4, face5,
        face6, face7, face8, face9, face10;
}

/// Construct the 4D Bezier control points for a given 5-point (penta-linear) element in spacetime.
/// This function interpolates both position and gradient information across the 5 vertices
/// of a 4D simplex (5-cell) to generate 35 Bezier control points representing the 4D Bezier simplex.
/// The resulting Bezier simplex can then be used for zero crossing check and distance check.
///
/// @param[in] p1–p5        The 4D vertex positions (x, y, z, t) of the 5-cell. Each is an Eigen::RowVector4d.
/// @param[in] v1–v5        The scalar function values (e.g., implicit field or signed distance) associated with each vertex.
/// @param[in] g1–g5        The gradients of the scalar field at each vertex, stored as 4D row vectors corresponding to ∂f/∂x, ∂f/∂y, ∂f/∂z, ∂f/∂t.
/// @param[out] bezier      An array of 35 Bezier control points representing the 4D Bezier simplex of the 5-cell.
///                         These control points encode both positional and derivative information to ensure
///                         smooth interpolation across the simplex.
/// @param[out] inside      A conservative tag indicating whether all Bezier control points are negative,
///                         implying that the entire 5-cell is inside the swept volume (no sign change within the simplex).
///
/// @return                 A boolean value that indicates whether the Bezier simplex has a zero crossing using sweep function

[[gnu::always_inline]]
inline bool bezier4D(
    const Eigen::RowVector4d& p1,
    const Eigen::RowVector4d& p2,
    const Eigen::RowVector4d& p3,
    const Eigen::RowVector4d& p4,
    const Eigen::RowVector4d& p5,
    const double v1,
    const double v2,
    const double v3,
    const double v4,
    const double v5,
    const Eigen::RowVector4d& g1,
    const Eigen::RowVector4d& g2,
    const Eigen::RowVector4d& g3,
    const Eigen::RowVector4d& g4,
    const Eigen::RowVector4d& g5,
    Eigen::RowVector<double, 35>& bezier,
    bool& inside)
{
   bezier4D(p1, p2, p3, p4, p5, v1, v2, v3, v4, v5, g1, g2, g3, g4, g5, bezier);

#ifndef no_inside
    inside =
        inside || bezier({0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 25, 26, 28, 31})
                          .maxCoeff() <= 0;
    inside = inside ||
             bezier({1, 2, 3, 4, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 24, 31, 32, 33, 34})
                     .maxCoeff() <= 0;
#endif

    // ───── final sign-test (unchanged logic) ──────────────────────────────────
    const double mx = bezier.maxCoeff();
    const double mn = bezier.minCoeff();
    return get_sign(mx) != get_sign(mn);
}

using MatrixX4dRowMajor =
    Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>;

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
    Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, 35, Eigen::RowMajor>> bezierCoords,
    std::vector<size_t>& domFIds);