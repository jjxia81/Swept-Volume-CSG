#pragma once

#include <Eigen/Core>
#include <numbers>

namespace csgf {

using Scalar = double;
const Scalar PI = M_PI;
using RVec3 = Eigen::RowVector3d;
using RVec4 = Eigen::RowVector4d;
using RVecX = Eigen::RowVectorXd;
using Mat3 = Eigen::Matrix3d;



inline Mat3 rotMX(double a) {
    Mat3 R;
    R << 1.0, 0.0,          0.0,
         0.0, std::cos(a), -std::sin(a),
         0.0, std::sin(a),  std::cos(a);
    return R;
}
// ==========================
// rotation
// ==========================
Mat3 rotMY(double a) {
    Mat3 R;
    R << std::cos(a), 0.0, std::sin(a),
         0.0,         1.0, 0.0,
        -std::sin(a), 0.0, std::cos(a);
    return R;
}

// ==========================
// dR/dt
// ==========================
Mat3 dRotMY_dt(double t) {
    double a = M_PI * t;
    double s = std::sin(a);
    double c = std::cos(a);

    Mat3 dR;
    dR << -s, 0.0,  c,
           0.0, 0.0, 0.0,
          -c, 0.0, -s;

    return M_PI * dR;
}

// ==========================
// trajectory
// ==========================
RVec3 trajLineX(double t) {
    return RVec3(0.5 * t, 0.0, 0.0);
}

// ==========================
// unified function
// ==========================
std::pair<double, RVec4>
sphere3d_generic(const RVec3& x,
          double t,
          const RVec3& shift,
          double scale)
{
    double circR = 0.55;

    RVec3 c  = trajLineX(2.0 * t);
    Mat3 R     = rotMY(M_PI * t);
    Mat3 dR    = dRotMY_dt(t);

    // v = (x - c) R - shift
    RVec3 v = (x - c) * R - shift;

    double value = v.dot(v) - circR * circR * scale;

    // ===== spatial gradient =====
    RVec3 grad_xyz = 2.0 * v * R.transpose();

    // ===== time derivative =====
    RVec3 dc_dt(1.0, 0.0, 0.0);

    RVec3 dv_dt = -dc_dt * R + (x - c) * dR;

    double grad_t = 2.0 * v.dot(dv_dt);

    RVec4 grad;
    grad << grad_xyz(0), grad_xyz(1), grad_xyz(2), grad_t;

    return {value, grad};
}


std::pair<Scalar, RVec4> sphere3d_f1(const RVec4& input)
{
    // f1 parameters
    RVec3 shift(0.2, 0.0, 0.0);
    double scale = 0.8;

    RVec3 x(input[0], input[1], input[2]);
    Scalar t = input[3];
    return sphere3d_generic(x, t, shift, scale);
}

std::pair<Scalar, RVec4> sphere3d_f2(const RVec4& input)
{
    // f2 parameters
    RVec3 shift(-0.4, 0.0, 0.0);
    double scale = 1.5;

    RVec3 x(input[0], input[1], input[2]);
    Scalar t = input[3];

    return sphere3d_generic(x, t, shift, scale);
}

inline std::pair<Scalar, size_t> csgf_sphere3d(const RVecX& input) {
    return input[0] >= input[1] ? std::make_pair(input[0], 0) : std::make_pair(input[1], 1);
}


// --- tetrahedron setup -------------------------------------------------------
//
// Mathematica uses SeedRandom[4] to add a RandomReal[{-0.3, 0.3}, 3] jitter to
// each of the 4 base vertices before the rotMX[0.5] tilt. That PRNG output
// isn't portable to C++. Two ways to handle this:
//
//   (A) Zero the jitter (default below). The tet is a clean tilted cube-diag
//       shape — same topology as Mathematica's, just without the randomness.
//
//   (B) Print the jittered pre-rotation vertices from Mathematica and paste:
//         Print[NumberForm[
//           Map[(# + RandomReal[{-.3,.3},3])&,
//               {{1,1,1},{-1,1,-1},{1,-1,-1},{-1,-1,1}}], 17]]
//       (evaluate right after SeedRandom[4]), then fill kBaseVertsJittered.

// Option A: paste the Mathematica `tetverts` output (post-jitter,
// post-rotMX[0.5], post-0.6 multiply) directly.
inline const std::array<RVec3, 4>& tetVerts() {
    static const std::array<RVec3, 4> v = {{
        RVec3( 0.532095,  0.857004,  0.267675  ),
        RVec3(-0.733447,  0.357681, -0.815757  ),
        RVec3( 0.458442, -0.793657, -0.0975139 ),
        RVec3(-0.674083, -0.196736,  0.67428   )
    }};
    return v;
}

// Option B (disabled): rebuild tetverts from base vertices in C++.
// Kept for reference; not used when Option A is active.
#if 0
inline const std::array<RVec3, 4>& kBaseVertsJittered() {
    static const std::array<RVec3, 4> v = {{
        RVec3( 1.0,  1.0,  1.0),
        RVec3(-1.0,  1.0, -1.0),
        RVec3( 1.0, -1.0, -1.0),
        RVec3(-1.0, -1.0,  1.0)
    }};
    return v;
}
inline const std::array<RVec3, 4>& tetVerts() {
    static const std::array<RVec3, 4> v = [] {
        const Mat3 Rx = rotMX(0.5);
        std::array<RVec3, 4> out{};
        const auto& base = kBaseVertsJittered();
        for (int i = 0; i < 4; ++i) out[i] = 0.6 * (base[i] * Rx);
        return out;
    }();
    return v;
}
#endif

// tetfaces = {{1,3,2},{1,2,4},{1,4,3},{2,3,4}}  (Mathematica 1-based).
inline constexpr std::array<std::array<int, 3>, 4> kTetFaceIdx = {{
    {{0, 2, 1}},  // {1,3,2}
    {{0, 1, 3}},  // {1,2,4}
    {{0, 3, 2}},  // {1,4,3}
    {{1, 2, 3}}   // {2,3,4}
}};

struct FaceData { RVec3 center; RVec3 normal; };

inline const std::array<FaceData, 4>& tetFaces() {
    static const std::array<FaceData, 4> data = [] {
        std::array<FaceData, 4> out{};
        const auto& V = tetVerts();
        for (int i = 0; i < 4; ++i) {
            const RVec3& a = V[kTetFaceIdx[i][0]];
            const RVec3& b = V[kTetFaceIdx[i][1]];
            const RVec3& c = V[kTetFaceIdx[i][2]];

            // center = Mean of the three face vertices
            out[i].center = (a + b + c) / 3.0;

            // normal = Normalize[ Cross[b - a, c - b] ]
            const RVec3 e1 = b - a;
            const RVec3 e2 = c - b;
            RVec3 n;
            n << e1(1)*e2(2) - e1(2)*e2(1),
                 e1(2)*e2(0) - e1(0)*e2(2),
                 e1(0)*e2(1) - e1(1)*e2(0);
            out[i].normal = n.normalized();
        }
        return out;
    }();
    return data;
}

// --- the four face functions -------------------------------------------------
//
// Direct translation of shapeCSGTetFk[x,y,z,t]:
//     p = ({x,y,z} - trajLineX[2 t]) . rotMY[t * Pi];
//     f = (p - tetcents[k]) . tetnorms[k];
//
// Chain-rule 4D gradient (d/dx, d/dy, d/dz, d/dt):
//
//   Let  c  = trajLineX[2 t] = (2 t, 0, 0)          => dc/dt = (2, 0, 0)
//        R  = rotMY(pi * t)                          => dR/dt given above
//        p  = (x - c) * R
//
//   df/dx_i = (n * R^T)_i         (p is linear in x)
//   df/dt   = dp/dt . n,
//     where   dp/dt = -(dc/dt) * R + (x - c) * (dR/dt)

inline std::pair<Scalar, RVec4>
tet_face_with_grad(int k, const RVec4& input)
{
    const Scalar x = input[0], y = input[1], z = input[2], t = input[3];

    const RVec3 xv(x, y, z);
    const RVec3 c  = trajLineX(2.0 * t);
    const Mat3  R  = rotMY(t * M_PI);
    const Mat3  dR = dRotMY_dt(t);

    const RVec3 p      = (xv - c) * R;
    const auto& F      = tetFaces()[k];
    const RVec3 offset = p - F.center;

    const Scalar value = offset.dot(F.normal);

    // spatial gradient: (n R^T)_i = sum_j n_j * R(i,j)
    RVec3 grad_xyz;
    grad_xyz << F.normal(0)*R(0,0) + F.normal(1)*R(0,1) + F.normal(2)*R(0,2),
                F.normal(0)*R(1,0) + F.normal(1)*R(1,1) + F.normal(2)*R(1,2),
                F.normal(0)*R(2,0) + F.normal(1)*R(2,1) + F.normal(2)*R(2,2);

    // time derivative: dp/dt . n
    const RVec3 dc_dt(2.0, 0.0, 0.0);
    const RVec3 dp_dt = (-dc_dt) * R + (xv - c) * dR;
    const Scalar grad_t = dp_dt.dot(F.normal);

    RVec4 grad(grad_xyz(0), grad_xyz(1), grad_xyz(2), grad_t);
    return {value, grad};
}

inline std::pair<Scalar, RVec4> tet_f1(const RVec4& input) { return tet_face_with_grad(0, input); }
inline std::pair<Scalar, RVec4> tet_f2(const RVec4& input) { return tet_face_with_grad(1, input); }
inline std::pair<Scalar, RVec4> tet_f3(const RVec4& input) { return tet_face_with_grad(2, input); }
inline std::pair<Scalar, RVec4> tet_f4(const RVec4& input) { return tet_face_with_grad(3, input); }

// --- CSG combination (intersection via max) ---------------------------------
//
// Mathematica tree decodes to:  max( max(f1,f2), max(f3,f4) )
inline std::pair<Scalar, size_t> csgf_tet(const RVecX& input) {
    size_t iA = input[0] >= input[1] ? 0 : 1;
    size_t iB = input[2] >= input[3] ? 2 : 3;
    Scalar sA = input[iA];
    Scalar sB = input[iB];
    return sA >= sB ? std::make_pair(sA, iA) : std::make_pair(sB, iB);
}

}