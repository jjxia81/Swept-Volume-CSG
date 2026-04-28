#pragma once

#include <cell_complex/CellComplex.h>
#include <cell_complex/generators.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <stdexcept>

namespace cell_complex {

template <int DIM>
CellComplex<DIM> load_uniform_grid(std::filesystem::path config_path)
{
    static_assert(DIM == 3 || DIM == 4, "Only 3D and 4D grids are supported.");
    YAML::Node config = YAML::LoadFile(config_path.string());
    if (config["grid"]) {
        std::array<int, DIM - 1> resolution;
        std::array<Scalar, DIM - 1> bbox_min;
        std::array<Scalar, DIM - 1> bbox_max;

        auto grid_config = config["grid"];
        if constexpr (DIM == 3) {
            resolution = {
                grid_config["resolution"][0].as<int>(),
                grid_config["resolution"][1].as<int>()};
            bbox_min = {
                grid_config["bbox_min"][0].as<Scalar>(),
                grid_config["bbox_min"][1].as<Scalar>()};
            bbox_max = {
                grid_config["bbox_max"][0].as<Scalar>(),
                grid_config["bbox_max"][1].as<Scalar>()};
        } else if constexpr (DIM == 4) {
            resolution = {
                grid_config["resolution"][0].as<int>(),
                grid_config["resolution"][1].as<int>(),
                grid_config["resolution"][2].as<int>()};
            bbox_min = {
                grid_config["bbox_min"][0].as<Scalar>(),
                grid_config["bbox_min"][1].as<Scalar>(),
                grid_config["bbox_min"][2].as<Scalar>()};
            bbox_max = {
                grid_config["bbox_max"][0].as<Scalar>(),
                grid_config["bbox_max"][1].as<Scalar>(),
                grid_config["bbox_max"][2].as<Scalar>()};
        }
        return cell_complex::make_column_grid<DIM - 1>(
            resolution,
            resolution[0], // nt = spatial resolution for uniform time sampling
            bbox_min,
            bbox_max);
    } else {
        throw std::runtime_error("Grid configuration not found in YAML file.");
    }
}

} // namespace cell_complex
