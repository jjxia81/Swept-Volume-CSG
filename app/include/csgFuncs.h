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

std::array<float, 3> bbox_min = {-0.5f, -0.75f, -0.75f};
std::array<float, 3> bbox_max = {1.5f, 0.75f, 0.75f};

}