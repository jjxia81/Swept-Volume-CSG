//
//  tet_quality.cpp
//  adaptive_mesh_refinement
//
//  Created by Yiwen Ju on 6/20/24.
//

#include "tet_quality.h"

///Below are functions used in `tet_radius_ratio`
///
/// returns the dot product of input arrays `a` and `b`.
double dot(const std::valarray<double>& a, const std::valarray<double>& b)
{
    return (a * b).sum();
}

/// returns the normalized vector of the input array `a` where it represents a vector in 3D.
std::valarray<double> normalize(const std::valarray<double>& a)
{
    return a / sqrt(dot(a, a));
}

/// returns the norm of the input array `a` where it represents a vector in 3D.
double norm(const std::valarray<double>& a)
{
    return sqrt(dot(a, a));
}

/// returns a perpendicular vector for the 2D vector `a`.
std::valarray<double> perp(const std::valarray<double>& a)
{
    return {-a[1], a[0]};
}

/// returns a cross product for the 3D vector `a` and `b`.
std::valarray<double> cross(const std::valarray<double>& a, const std::valarray<double>& b)
{
    std::valarray<double> c(3);
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
    return c;
}

double tet_radius_ratio(const std::array<std::valarray<double>, 4>& pts)
{
    // Determine side vectors
    std::array<std::valarray<double>, 6> side;
    for (int i = 0; i < 6; ++i) {
        side[i].resize(3);
    }
    side[0] = {pts[1][0] - pts[0][0], pts[1][1] - pts[0][1], pts[1][2] - pts[0][2]};

    side[1] = {pts[2][0] - pts[1][0], pts[2][1] - pts[1][1], pts[2][2] - pts[1][2]};

    side[2] = {pts[0][0] - pts[2][0], pts[0][1] - pts[2][1], pts[0][2] - pts[2][2]};

    side[3] = {pts[3][0] - pts[0][0], pts[3][1] - pts[0][1], pts[3][2] - pts[0][2]};

    side[4] = {pts[3][0] - pts[1][0], pts[3][1] - pts[1][1], pts[3][2] - pts[1][2]};

    side[5] = {pts[3][0] - pts[2][0], pts[3][1] - pts[2][1], pts[3][2] - pts[2][2]};

    std::valarray<double> numerator = dot(side[3], side[3]) * cross(side[2], side[0]) +
                                      dot(side[2], side[2]) * cross(side[3], side[0]) +
                                      dot(side[0], side[0]) * cross(side[3], side[2]);

    double area_sum;
    area_sum = (norm(cross(side[2], side[0])) + norm(cross(side[3], side[0])) +
                norm(cross(side[4], side[1])) + norm(cross(side[3], side[2]))) *
               0.5;

    double volume = dot(pts[0] - pts[3], cross(pts[1] - pts[3], pts[2] - pts[3])) / 6;
    const double radius_ratio = (108 * volume * volume) / (norm(numerator) * area_sum);
    return radius_ratio;
}


double tet_radius_ratio(const std::array<Eigen::RowVector3d, 4>& pts)
{
    Eigen::RowVector3d s0 = pts[1] - pts[0];
    Eigen::RowVector3d s1 = pts[2] - pts[1];
    Eigen::RowVector3d s2 = pts[0] - pts[2];
    Eigen::RowVector3d s3 = pts[3] - pts[0];
    Eigen::RowVector3d s4 = pts[3] - pts[1];
    Eigen::RowVector3d s5 = pts[3] - pts[2];

    // Numerator vector (circumradius numerator)
    Eigen::RowVector3d numerator =
        s3.dot(s3) * s2.cross(s0) +
        s2.dot(s2) * s3.cross(s0) +
        s0.dot(s0) * s3.cross(s2);

    // Face areas
    double area_sum = (s2.cross(s0).norm() +
                       s3.cross(s0).norm() +
                       s4.cross(s1).norm() +
                       s3.cross(s2).norm()) * 0.5;

    // Signed volume
    double volume = (pts[0] - pts[3]).dot(
                        (pts[1] - pts[3]).cross(pts[2] - pts[3])) / 6.0;

    if (std::abs(volume) < 1e-14 || area_sum < 1e-14) return 0.0;

    const double radius_ratio = (108.0 * volume * volume) / (numerator.norm() * area_sum);
    return radius_ratio;
}