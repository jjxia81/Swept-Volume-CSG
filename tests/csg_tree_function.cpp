#include <catch2/catch.hpp>
#include "stf/yaml_parser.h"
#include <array>
#include <string>
#include <iostream>
#include <cmath>

using Scalar = double;
constexpr int dim = 3;

// ── helpers ───────────────────────────────────────────────────────────────────

TEST_CASE("twoSphere3d CSG - value and gradient vs Mathematica reference", "[stf][csg]") {
    std::string path = std::string(TEST_FILE) + "/csg/twoSphere3d.yaml";
    auto owned = stf::YamlParser<3>::parse_csg_from_file(path);
    stf::CSGTree<3>* func = owned.get();  // context stays alive inside owned
    // ── test points {x, y, z, t} ─────────────────────────────────────────────
    const std::vector<std::array<Scalar, dim + 1>> test_points = {
        {0.0,  0.0,  0.0,  0.0},   // inside both
        {0.0,  0.2,  0.0,  0.0},   // inside both, y-offset
        {0.5,  0.0,  0.0,  0.0},   // inside ball1 only
        {-0.6, 0.0,  0.0,  0.0},   // inside ball2 only
        {2.0,  0.0,  0.0,  0.0},   // outside both
        {0.0,  0.0,  0.0,  0.25},  // t=0.25 origin
        {0.0,  0.1,  0.1,  0.25},  // t=0.25 yz offset
        {-0.5, 0.0,  0.0,  0.25},  // t=0.25 -x region
        {0.0,  0.0,  0.0,  0.5},   // t=0.5 origin
        {0.0,  0.2,  0.0,  0.5},   // t=0.5 y-offset
        {0.0,  0.0,  0.2,  0.5},   // t=0.5 z-offset
        {5.0,  5.0,  5.0,  0.5},   // t=0.5 far outside
        {0.0,  0.0,  0.0,  0.75},  // t=0.75 origin
        {-0.5, 0.1,  0.0,  0.75},  // t=0.75 -x y-offset
        {-0.5, 0.0,  0.1,  0.75},  // t=0.75 -x z-offset
        {-0.9, 0.0,  0.0,  1.0},   // t=1 inside both transformed
        {-0.9, 0.1,  0.1,  1.0},   // t=1 inside both yz-offset
        {-1.5, 0.0,  0.0,  1.0},   // t=1 inside ball1 final only
        {-0.2, 0.0,  0.0,  1.0},   // t=1 inside ball2 final only
        {5.0,  0.0,  0.0,  1.0},   // t=1 outside both
    };

    // ── Mathematica reference values: ref_values[func_idx][point_idx] ────────
    const std::vector<std::vector<Scalar>> ref_values = {
        // f1 (ball1 sweep)
        {-0.291935, -0.209092, -0.191935,  0.308065,  1.30807,
         -0.0757491,-0.0213025, 0.410635,  0.0465815,  0.0825213,
          0.148377,  8.01042,   0.132859,  0.630093,   0.642627,
          1.20807,   1.21394,   1.80807,   0.508065,   3.70807},
        // f2 (ball2 sweep)
        {-0.27361,  -0.226396,  0.22639,  -0.47361,   1.72639,
         -0.388867, -0.462636, -0.1275,   -0.0332973, -0.00278929,
         -0.135093,  7.47562,   0.397261,  0.888314,   0.8701,
          1.62639,   1.63073,   2.22639,   0.92639,    2.92639},
    };

    // ── Mathematica reference gradients: ref_grads[func_idx][point_idx] ──────
    const std::vector<std::vector<std::array<Scalar, dim + 1>>> ref_grads = {
        // f1 gradients {df/dx, df/dy, df/dz, df/dt}
        {
            {-1.,        0.,        0.,        1.       },
            {-0.707107,  0.707107,  0.,        0.707107 },
            { 1.,        0.,        0.,       -1.       },
            {-1.,        0.,        0.,        1.       },
            { 1.,        0.,        0.,       -1.       },
            {-0.940497,  0.,        0.339803,  0.673616 },
            {-0.831692,  0.21248,   0.512972,  0.690089 },
            {-0.987648,  0.,        0.156687,  0.618462 },
            {-0.928477,  0.,        0.371391,  0.345098 },
            {-0.870388,  0.348155,  0.348155,  0.323507 },
            {-0.780869,  0.,        0.624695,  0.290234 },
            { 0.529265,  0.588073,  0.611595, -0.196718 },
            {-0.974046,  0.,        0.226349,  0.440725 },
            {-0.988013,  0.0891243, 0.126041,  0.493052 },
            {-0.977098,  0.,        0.212788,  0.448446 },
            {-1.,        0.,        0.,        1.       },
            {-0.996558,  0.058621,  0.058621,  0.959725 },
            {-1.,        0.,        0.,        1.       },
            {-1.,        0.,        0.,        1.       },
            { 1.,        0.,        0.,       -1.       },
        },
        // f2 gradients {df/dx, df/dy, df/dz, df/dt}
        {
            { 1.,        0.,        0.,       -1.       },
            { 0.894427,  0.447214,  0.,       -0.894427 },
            { 1.,        0.,        0.,       -1.       },
            {-1.,        0.,        0.,        1.       },
            { 1.,        0.,        0.,       -1.       },
            { 0.115342,  0.,       -0.993326,  0.664815 },
            { 0.155672,  0.473992, -0.866659,  0.476095 },
            {-0.855427,  0.,       -0.517923,  2.07575  },
            {-0.780869,  0.,       -0.624695,  1.76214  },
            {-0.745356,  0.298142, -0.596285,  1.682    },
            {-0.928477,  0.,       -0.371391,  2.09523  },
            { 0.552199,  0.613555,  0.56447,  -1.24611  },
            {-0.964489,  0.,       -0.264124,  1.58682  },
            {-0.981381,  0.0640236,-0.181086,  1.6925   },
            {-0.992961,  0.,       -0.118444,  1.77004  },
            {-1.,        0.,        0.,        1.       },
            {-0.998115,  0.0433963, 0.0433963, 1.05265  },
            {-1.,        0.,        0.,        1.       },
            {-1.,        0.,        0.,        1.       },
            { 1.,        0.,        0.,       -1.       },
        },
    };

    const Scalar val_tol  = 1e-4;
    const Scalar grad_tol = 1e-4;
    const char* labels[]  = {"df/dx", "df/dy", "df/dz", "df/dt"};

    SECTION("subfunction values vs Mathematica") {
        int root = func->get_root_index();
        auto [left_idx, right_idx] = func->get_children(root);

        // std::cout << " number of nodes " << func->get_num_nodes() << std::endl;

        std::array<int, 2> leaf_indices = {left_idx, right_idx};

        for (int fi = 0; fi < 2; fi++) {
            for (size_t j = 0; j < test_points.size(); j++) {
                DYNAMIC_SECTION("f" << fi+1 << " value at point " << j) {
                    const auto& pt = test_points[j];
                    std::array<Scalar, dim> pos = {pt[0], pt[1], pt[2]};
                    Scalar t = pt[3];

                    Scalar val = func->value_at(leaf_indices[fi], pos, t);
                    Scalar ref = ref_values[fi][j];

                    INFO("point: {" << pt[0] << "," << pt[1] << ","
                                << pt[2] << "," << pt[3] << "}");
                    INFO("computed=" << val << " ref=" << ref);
                    CHECK(std::abs(val - ref) < val_tol);
                }
            }
        }
    }

    SECTION("subfunction gradients vs Mathematica") {
        int root = func->get_root_index();
        auto [left_idx, right_idx] = func->get_children(root);
        std::array<int, 2> leaf_indices = {left_idx, right_idx};

        for (int fi = 0; fi < 2; fi++) {
            for (size_t j = 0; j < test_points.size(); j++) {
                DYNAMIC_SECTION("f" << fi+1 << " gradient at point " << j) {
                    const auto& pt = test_points[j];
                    std::array<Scalar, dim> pos = {pt[0], pt[1], pt[2]};
                    Scalar t = pt[3];

                    auto grad = func->gradient_at(leaf_indices[fi], pos, t);
                    const auto& ref = ref_grads[fi][j];

                    INFO("point: {" << pt[0] << "," << pt[1] << ","
                                << pt[2] << "," << pt[3] << "}");
                    for (int k = 0; k < dim + 1; k++) {
                        INFO(labels[k] << " computed=" << grad[k]
                                    << " ref=" << ref[k]
                                    << " diff=" << std::abs(grad[k] - ref[k]));
                        CHECK(std::abs(grad[k] - ref[k]) < grad_tol);
                    }
                }
            }
        }
    }
}


TEST_CASE("fourSphere3d CSG - value and gradient vs Mathematica reference", "[stf][csg]") {
    std::string path = std::string(TEST_FILE) + "/csg/tet3d.yaml";

    auto owned = stf::YamlParser<3>::parse_csg_from_file(path);
    stf::CSGTree<3>* func = owned.get();

    const std::vector<std::array<Scalar, dim + 1>> test_points = {
        {0.0,  0.0,  0.0,  0.0},
        {0.0,  0.2,  0.0,  0.0},
        {0.5,  0.0,  0.0,  0.0},
        {-0.6, 0.0,  0.0,  0.0},
        {2.0,  0.0,  0.0,  0.0},
        {0.0,  0.0,  0.0,  0.25},
        {0.0,  0.1,  0.1,  0.25},
        {-0.5, 0.0,  0.0,  0.25},
        {0.0,  0.0,  0.0,  0.5},
        {0.0,  0.2,  0.0,  0.5},
        {0.0,  0.0,  0.2,  0.5},
        {5.0,  5.0,  5.0,  0.5},
        {0.0,  0.0,  0.0,  0.75},
        {-0.5, 0.1,  0.0,  0.75},
        {-0.5, 0.0,  0.1,  0.75},
        {-0.9, 0.0,  0.0,  1.0},
        {-0.9, 0.1,  0.1,  1.0},
        {-1.5, 0.0,  0.0,  1.0},
        {-0.2, 0.0,  0.0,  1.0},
        {5.0,  0.0,  0.0,  1.0},
    };
const std::vector<std::vector<Scalar>> ref_values = {
        // f1
        {-0.240306, -0.211271,  0.0645789, -0.606167, 0.979233,
         -0.180413, -0.264109, -0.120521,   0.538864,  0.567899,
          0.416911, -5.67996,   1.23289,    1.73847,   1.73593,
          1.52803,   1.62046,   1.89389,    1.10119,  -2.06961},
        // f2
        {-0.437626, -0.284514, -0.720222, -0.0985107, -1.56801,
         -0.346472, -0.208216, -0.255318, -0.744995,  -0.591883,
         -0.631957,  7.44561,  -1.36312,  -1.59506,   -1.65338,
         -2.07668,  -2.03086,  -2.4158,   -1.68105,    1.25795},
        // f3
        {-0.300107, -0.342194, -0.0639209, -0.583529,  0.644636,
         -0.769725, -0.763648, -1.23934,   -1.15602,  -1.1981,
         -1.25049,  -0.290509, -0.706911,  -0.863556, -0.936436,
          1.06977,   0.963136,  1.35319,    0.73911,  -1.71722},
        // f4
        {-0.367649, -0.521293, -0.659575, -0.0173372, -1.53535,
         -0.0683892,-0.122494,  0.23087,  -0.105068,  -0.258712,
          0.0117029,-2.33983,  -0.708408, -0.898817,  -0.762143,
         -2.06082,  -2.11139,  -2.41113,  -1.65212,    1.38391},
    };

    const std::vector<std::vector<std::array<Scalar, dim + 1>>> ref_grads = {
        // f1 gradients
        {
            { 0.609769,  0.145174, -0.77917,  -1.21954  },
            { 0.609769,  0.145174, -0.77917,  -1.21954  },
            { 0.609769,  0.145174, -0.77917,  -2.44346  },
            { 0.609769,  0.145174, -0.77917,   0.249163 },
            { 0.609769,  0.145174, -0.77917,  -6.11521  },
            {-0.119785,  0.145174, -0.982128,  1.78229  },
            {-0.119785,  0.145174, -0.982128,  1.81992  },
            {-0.119785,  0.145174, -0.982128,  3.32502  },
            {-0.77917,   0.145174, -0.609769,  3.47399  },
            {-0.77917,   0.145174, -0.609769,  3.47399  },
            {-0.77917,   0.145174, -0.609769,  3.96355  },
            {-0.77917,   0.145174, -0.609769,  6.13493  },
            {-0.982128,  0.145174,  0.119785,  1.39979  },
            {-0.982128,  0.145174,  0.119785,  1.21163  },
            {-0.982128,  0.145174,  0.119785,  1.52017  },
            {-0.609769,  0.145174,  0.77917,  -5.87918  },
            {-0.609769,  0.145174,  0.77917,  -5.68762  },
            {-0.609769,  0.145174,  0.77917,  -7.34788  },
            {-0.609769,  0.145174,  0.77917,  -4.1657   },
            {-0.609769,  0.145174,  0.77917,   8.56304  },
        },
        // f2 gradients
        {
            {-0.565192,  0.76556,  0.307369,  1.13038  },
            {-0.565192,  0.76556,  0.307369,  1.13038  },
            {-0.565192,  0.76556,  0.307369,  1.6132   },
            {-0.565192,  0.76556,  0.307369,  0.551008 },
            {-0.565192,  0.76556,  0.307369,  3.06164  },
            {-0.182309,  0.76556,  0.616994, -0.604555 },
            {-0.182309,  0.76556,  0.616994, -0.547281 },
            {-0.182309,  0.76556,  0.616994, -1.57373  },
            { 0.307369,  0.76556,  0.565192, -2.39034  },
            { 0.307369,  0.76556,  0.565192, -2.39034  },
            { 0.307369,  0.76556,  0.565192, -2.58347  },
            { 0.307369,  0.76556,  0.565192,  1.65954  },
            { 0.616994,  0.76556,  0.182309, -2.0931   },
            { 0.616994,  0.76556,  0.182309, -2.37947  },
            { 0.616994,  0.76556,  0.182309, -2.5733   },
            { 0.565192,  0.76556, -0.307369,  1.66994  },
            { 0.565192,  0.76556, -0.307369,  1.49238  },
            { 0.565192,  0.76556, -0.307369,  2.24931  },
            { 0.565192,  0.76556, -0.307369,  0.993998 },
            { 0.565192,  0.76556, -0.307369, -4.02727  },
        },
        // f3 gradients
        {
            { 0.472371, -0.210437,  0.85591,  -0.944743 },
            { 0.472371, -0.210437,  0.85591,  -0.944743 },
            { 0.472371, -0.210437,  0.85591,   0.399717 },
            { 0.472371, -0.210437,  0.85591,  -2.55809  },
            { 0.472371, -0.210437,  0.85591,   4.4331   },
            { 0.939237, -0.210437,  0.271203, -2.30448  },
            { 0.939237, -0.210437,  0.271203, -2.59955  },
            { 0.939237, -0.210437,  0.271203, -2.73048  },
            { 0.85591,  -0.210437, -0.472371, -0.227821 },
            { 0.85591,  -0.210437, -0.472371, -0.227821 },
            { 0.85591,  -0.210437, -0.472371, -0.765605 },
            { 0.85591,  -0.210437, -0.472371,-21.0924   },
            { 0.271203, -0.210437, -0.939237,  3.88364  },
            { 0.271203, -0.210437, -0.939237,  5.35899  },
            { 0.271203, -0.210437, -0.939237,  5.27379  },
            {-0.472371, -0.210437, -0.85591,   8.74261  },
            {-0.472371, -0.210437, -0.85591,   8.89101  },
            {-0.472371, -0.210437, -0.85591,  10.356    },
            {-0.472371, -0.210437, -0.85591,   6.86037  },
            {-0.472371, -0.210437, -0.85591,  -7.12202  },
        },
        // f4 gradients
        {
            {-0.583852, -0.768224, -0.262581,  1.1677   },
            {-0.583852, -0.768224, -0.262581,  1.1677   },
            {-0.583852, -0.768224, -0.262581,  0.755244 },
            {-0.583852, -0.768224, -0.262581,  1.66266  },
            {-0.583852, -0.768224, -0.262581, -0.482141 },
            {-0.598519, -0.768224,  0.227173,  0.840195 },
            {-0.598519, -0.768224,  0.227173,  1.02823  },
            {-0.598519, -0.768224,  0.227173,  0.483352 },
            {-0.262581, -0.768224,  0.583852, -1.30906  },
            {-0.262581, -0.768224,  0.583852, -1.30906  },
            {-0.262581, -0.768224,  0.583852, -1.14408  },
            {-0.262581, -0.768224,  0.583852,  11.9867  },
            { 0.227173, -0.768224,  0.598519, -3.2748   },
            { 0.227173, -0.768224,  0.598519, -4.21495  },
            { 0.227173, -0.768224,  0.598519, -4.28632  },
            { 0.583852, -0.768224,  0.262581, -3.55998  },
            { 0.583852, -0.768224,  0.262581, -3.7434   },
            { 0.583852, -0.768224,  0.262581, -4.05493  },
            { 0.583852, -0.768224,  0.262581, -2.98254  },
            { 0.583852, -0.768224,  0.262581,  1.30706  },
        },
    };

    const Scalar val_tol  = 1e-4;
    const Scalar grad_tol = 1e-4;
    const char* labels[]  = {"df/dx", "df/dy", "df/dz", "df/dt"};

    // Collect all leaf node indices via post-order traversal
    std::vector<int> leaf_indices;
    func->visit_postorder([&](stf::CSGTree<3>::NodeInfo info) {
        if (info.is_leaf()) {
            leaf_indices.push_back(info.node_index);
            // std::cout << " lead node id " << info.node_index <<std::endl;
        }
    });

    auto pt = test_points[0];
    std::array<Scalar, dim> pos = {pt[0], pt[1], pt[2]};
                    Scalar t = pt[3];

    Scalar val = func->value_at(leaf_indices[0], pos, t);
    Scalar ref = ref_values[leaf_indices[0]][0];

    // INFO("point: {" << pt[0] << "," << pt[1] << ","
    //                                << pt[2] << "," << pt[3] << "}");
    // std::cout << " csg " << leaf_indices[0] << "  , val " << val << std::endl;
    // std::cout << " ref  val " << val << std::endl;
    // return;

    REQUIRE(leaf_indices.size() == 4);

   
    SECTION("subfunction values vs Mathematica") {
        for (int fi = 0; fi < 4; fi++) {
            for (size_t j = 0; j < test_points.size(); j++) {
                // DYNAMIC_SECTION("f" << fi+1 << " value at point " << j) 
                {
                    const auto& pt = test_points[j];
                    std::array<Scalar, dim> pos = {pt[0], pt[1], pt[2]};
                    Scalar t = pt[3];

                    Scalar val = func->value_at(leaf_indices[fi], pos, t);
                    Scalar ref = ref_values[fi][j];
                    Scalar diff = std::abs(val - ref);

                    // Always print so you see all results, not just failures
                    if (diff >= val_tol)
                    {
                        std::cout << "f" << fi+1 
                            << " pt" << j 
                            << " {" << pt[0] << "," << pt[1] << "," << pt[2] << "," << pt[3] << "}"
                            << " val=" << val 
                            << " ref=" << ref 
                            << " diff=" << diff
                            << (diff < val_tol ? " OK" : " FAIL")
                            << std::endl;

                    }
                    INFO("point: {" << pt[0] << "," << pt[1] << "," << pt[2] << "," << pt[3] << "}");
                    INFO("computed=" << val << " ref=" << ref << " diff=" << diff);
                    CHECK(diff < val_tol);
                }
            }
        }
    }
    SECTION("subfunction gradients vs Mathematica") {
        for (int fi = 0; fi < 4; fi++) {
            for (size_t j = 0; j < test_points.size(); j++) {
                DYNAMIC_SECTION("f" << fi+1 << " gradient at point " << j) {
                    const auto& pt = test_points[j];
                    std::array<Scalar, dim> pos = {pt[0], pt[1], pt[2]};
                    Scalar t = pt[3];

                    auto grad = func->gradient_at(leaf_indices[fi], pos, t);
                    const auto& ref = ref_grads[fi][j];

                    // std::cout << "f" << fi+1 
                    //         << " pt" << j
                    //         << " {" << pt[0] << "," << pt[1] << "," << pt[2] << "," << pt[3] << "}"
                    //         << std::endl;

                    bool all_ok = true;
                    for (int k = 0; k < dim + 1; k++) {
                        Scalar diff = std::abs(grad[k] - ref[k]);
                        bool ok = diff < grad_tol;
                        if (!ok) 
                        {
                            all_ok = false;
                            std::cout << "  " << labels[k] 
                                    << " computed=" << grad[k] 
                                    << " ref=" << ref[k] 
                                    << " diff=" << diff
                                    << (ok ? " OK" : " FAIL")
                                    << std::endl;
                            
                        }
                        INFO(labels[k] << " computed=" << grad[k]
                                        << " ref=" << ref[k]
                                        << " diff=" << diff);
                        CHECK(diff < grad_tol);
                    }
                }
            }
        }
    }
}