// csg_tet.h — direct C++ translation of the Mathematica f1..f4.
//
// Mathematica source (for reference):
//
//   rotMY[a_] := {{Cos[a],0,Sin[a]},{0,1,0},{-Sin[a],0,Cos[a]}};
//   rotMX[a_] := {{1,0,0},{0,Cos[a],-Sin[a]},{0,Sin[a],Cos[a]}};
//   trajLineX[t_] := t*{1,0,0};
//   SeedRandom[4];
//   tetverts = 0.6 * (Map[(# + RandomReal[{-.3,.3},3])&,
//                         {{1,1,1},{-1,1,-1},{1,-1,-1},{-1,-1,1}}] . rotMX[0.5]);
//   tetfaces = {{1,3,2},{1,2,4},{1,4,3},{2,3,4}};
//   tetcents = Map[Mean[tetverts[[#]]]&, tetfaces];
//   tetnorms = Map[Normalize[Cross[v#[[2]]-v#[[1]], v#[[3]]-v#[[2]]]]&, tetfaces];
//   shapeCSGTetFk[x,y,z,t] :=
//       Module[{p}, p = ({x,y,z} - trajLineX[2 t]) . rotMY[t Pi];
//                   (p - tetcents[[k]]) . tetnorms[[k]]];

#pragma once

#include <Eigen/Core>
#include <array>
#include <cmath>

namespace csgfTet {

using Scalar = double;
using RVec3  = Eigen::RowVector3d;
using RVec4  = Eigen::RowVector4d;
using Mat3   = Eigen::Matrix3d;
using RVecX = Eigen::RowVectorXd;

// --- rotations ---------------------------------------------------------------

// inline Mat3 rotMY(double a) {
//     Mat3 R;
//     R <<  std::cos(a), 0.0, std::sin(a),
//           0.0,         1.0, 0.0,
//          -std::sin(a), 0.0, std::cos(a);
//     return R;
// }

// inline Mat3 rotMX(double a) {
//     Mat3 R;
//     R << 1.0, 0.0,          0.0,
//          0.0, std::cos(a), -std::sin(a),
//          0.0, std::sin(a),  std::cos(a);
//     return R;
// }

// --- trajectory --------------------------------------------------------------

inline RVec3 trajLineX(double t) {
    return RVec3(t, 0.0, 0.0);
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

// --- derivative of the rotation matrix ---------------------------------------
//   R(t) = rotMY(pi * t)
//   dR/dt = pi * {{-sin, 0, cos}, {0, 0, 0}, {-cos, 0, -sin}}   (arg pi*t)

// inline Mat3 dRotMY_dt(double t) {
//     const double a = M_PI * t;
//     const double s = std::sin(a);
//     const double c = std::cos(a);
//     Mat3 dR;
//     dR << -M_PI * s,  0.0,       M_PI * c,
//            0.0,       0.0,       0.0,
//           -M_PI * c,  0.0,      -M_PI * s;
//     return dR;
// }

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
// For max(a,b) the gradient is the gradient of the larger operand.

inline std::pair<Scalar, RVec4>
csg_max(const std::pair<Scalar, RVec4>& a,
        const std::pair<Scalar, RVec4>& b)
{
    return (a.first >= b.first) ? a : b;
}

// inline std::pair<Scalar, RVec4> csgf_tet(const RVec4& input) {
//     auto A = csg_max(tet_f1(input), tet_f2(input));
//     auto B = csg_max(tet_f3(input), tet_f4(input));
//     return csg_max(A, B);
// }

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

std::array<float, 3> bbox_min = {-1.0f, -1.0f, -1.0f};
std::array<float, 3> bbox_max = {3.0f, 1.0f, 1.2f};


} // namespace csgf