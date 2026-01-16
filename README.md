# Lifted Surfacing of Generalized Sweep Volumes

This code implements the ACM SIGGRAPH ASIA 2025 paper: Lifted Surfacing of Generalized Sweep Volumes

<img width="1600" alt="ball-rolling" src="https://github.com/user-attachments/assets/4751b437-091b-4626-976e-3c0a74132838" />

>Top: A wire-like ball rolls forward while offsetting, changing its genus from 41 to 29. Bottom: The time-colored sweep boundary and sharp creases (2nd row), in transparency (3rd row, creases hidden for clarity), and a cut-away view (bottom row).

Given any sweep represented as a smooth time-varying implicit function satisfying a genericity assumption, this algorithm produces a watertight and intersection-free surface that faithfully approximates the geometric and topological features.

## Build

### Dependencies

All third-party libraries are open-source and automatically fetched using CMake.

### C++ Build

Use the following command to build:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```
The C++ library `libsweep` and the command line tool `generalized_sweep` will be generated in the
build directory.

### Python Bindings

We also provide a Python binding for easy integration with Python workflows:

```bash
# Install the Python package
pip install sweep3d
```

## Usage

### C++ API

The C++ API consists of two simple functions, `generalized_sweep` and
`generalized_sweep_from_config` defined in `include/sweep/generalized_sweep.h`.
For complete API documentation, please see [generalized_sweep.h](include/sweep/generalized_sweep.h).

#### `generalized_sweep`

The `generalized_sweep` function provides the most general API. It takes the following inputs:

* a user-provided space-time function for pointwise function and gradient evaluation,
* (optional) a simple initial grid specification ([doc](#grid-parameters)), and
* (optional) customized sweep parameters ([doc](#sweep-options))

and generates the following outputs:

* Sweep surface (final output)
* Sweep envelope
* Envelope arrangement

In addition to the sweep surface, our code also provides the sweep envelope and envelope
arrangement for advanced users. Please see our paper for their definitions.

Here is a simple example:

```c++
#include <sweep/generalized_sweep.h>
#include <lagrange/io/save_mesh.h> // For IO only

// Define implicit space-time function
// Sweeping a ball of radius `r` along the X axis by a distance `d`.
sweep::SpaceTimeFunc f = [](EigenRowVector4d p) {
    constexpr double r = 0.2;
    constexpr double d = 0.5;

    double x = p[0];
    double y = p[1];
    double z = p[2];
    double t = p[3];

    double val = (x + d * t) * (x + d * t) + y * y + z * z - r * r;
    Eigen::RowVector4d grad {
        2 * (x + d * t),
        2 * y,
        2 * z,
        2 * d * (x + d * t)
    };
    return {val, grad};
};

// Define an initial coarse grid
sweep::GridSpec grid;
grid.bbox_min = {-0.3, -0.3, -0.3};
grid.bbox_max = {0.8, 0.8, 0.8};

// Compute the sweep surface
auto r = sweep::generalized_sweep(f, grid);
lagrange::io::save_mesh("sweep_surface.obj", r.sweep_surface);
```

#### `generalized_sweep_from_config`

The `generalized_sweep_from_config` function loads two configuration files:
* `function_file` defines the space-time function
* `config_file` defines the initial grid spec and sweep parameters

It outputs the same set of meshes as `generalized_sweep`. Both `function_file` and `config_file`
are in YAML format.


The function file is a YAML file that defines a space-time function supported by the [space-time-functions](https://github.com/adobe-research/space-time-functions) library.
Here is a simple function file that sweeps a ball along the X axis. Please see the [spec](https://github.com/adobe-research/space-time-functions/blob/main/doc/yaml_spec.md) for a complete set of supported transforms and shapes.

```yaml
type: sweep
dimension: 3

# The base shape
primitive:
  type: ball
  center: [0.0, 0.0, 0.0]
  radius: 0.2
  degree: 2

# Sweeping trajectory
transform:
  type: polyline
  points:
  - [-0.5, 0.0, 0.0]
  - [0.5, 0.0, 0.0]
```

The `config_file` is used to specify the initial grid and the sweep options. Here is an example:

```yaml
grid:
  resolution: [4, 4, 4]
  bbox_min: [-1, -1, -1]
  bbox_max: [1, 1, 1]

parameters:
  epsilon_env: 5e-4
  epsilon_sil: 5e-4
  with_snapping: false
  with_insideness_check: true
```

The parameters section will be used to construct the `sweep::SweepOptions`.

Here is an example:

```c++
#include <sweep/generalized_sweep.h>
#include <lagrange/io/save_mesh.h> // For IO only

auto r = sweep::generalized_sweep_from_config(
    "example/simple/sweep.yaml",
    "example/simple/config.yaml"
);
lagrange::io::save_mesh("sweep_surface.obj", r.sweep_surface);
```

Please see more config-files-based examples in the [example](example) folder.

### Python API

```python
import numpy as np
import sweep3d
import lagrange # For IO only

# Space-time function
def my_space_time_function(point_4d):
    """Space-time implicit function.
    
    :param point_4d: 4D point [x, y, z, t] where t ∈ [0, 1]
    :type point_4d: numpy.ndarray
    :return: Tuple of (value, gradient)
    :rtype: tuple(float, numpy.ndarray)
    """
    x, y, z, t = point_4d
    r = 0.2
    d = 0.5
    value = (x - t * d)**2 + y**2 + z**2 - r**2
    gradient = np.array([2 * (x - t * d), 2 * y, 2 * z, -2 * d * (x - t * d)])
    return (value, gradient)

# Initial grid
grid_spec = sweep3d.GridSpec()
grid_spec.bbox_min = [-0.3, -0.3, -0.3]
grid_spec.bbox_max = [0.8, 0.8, 0.8]

# Compute sweep surface
result = sweep3d.generalized_sweep(my_space_time_function, grid_spec)

# Extract mesh data
V = result.sweep_surface.vertices
F = result.sweep_surface.facets
print(f"#Vertices: {len(V)}, #Faces: {len(F)}")

# Save mesh
lagrange.io.save_mesh("out.msh", result.sweep_surface)
```

Please see `help(sweep3d)` for more details.


## Grid Parameters

The `GridSpec` struct defines the initial spatial grid used for sweep computation. The grid represents the 3D spatial domain (the time dimension is handled separately).

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resolution` | `[int, int, int]` | `[4, 4, 4]` | Number of grid cells in the x, y, z directions. Higher resolution provides better initial sampling but increases computation time. |
| `bbox_min` | `[float, float, float]` | `[-0.2, -0.2, -0.2]` | Minimum corner of the axis-aligned bounding box that encloses the sweep volume. |
| `bbox_max` | `[float, float, float]` | `[1.2, 1.2, 1.2]` | Maximum corner of the axis-aligned bounding box that encloses the sweep volume. |

**Note:** The bounding box should be large enough to fully enclose the swept volume throughout the entire trajectory (t ∈ [0, 1]).

## Sweep Options

The `SweepOptions` struct provides fine-grained control over the sweep computation. All parameters are optional and have sensible defaults.

### Refinement Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epsilon_env` | `double` | `5e-4` | Tolerance for envelope approximation. |
| `epsilon_sil` | `double` | `5e-3` | Tolerance for silhouette set approximation. |
| `max_split` | `int` | unlimited | Maximum number of splits allowed during grid refinement. |
| `with_adaptive_refinement` | `bool` | `true` | Enable/disable adaptive grid refinement. |
| `initial_time_samples` | `size_t` | `8` | Number of initial uniform time samples per spatial grid vertex. |

### Quality Control Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_tet_radius_ratio` | `double` | `1e-5` | Minimum acceptable tetrahedron in-radius to circum-radius ratio during grid refinement. Tets below this threshold will not be refined further. |
| `min_tet_edge_length` | `double` | `2e-5` | Minimum acceptable tetrahedron edge length during grid refinement. Tets with longest edge below this threshold will not be refined further. |

### Surface Extraction Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `with_insideness_check` | `bool` | `true` | Whether to perform insideness checks during grid refinement. If true, the algorithm will stop refinement early once it detects a cell is inside the swept volume. |
| `with_snapping` | `bool` | `true` | Whether to enable vertex snapping during isocontouring. Improves the quality of the extracted mesh. |
| `volume_threshold` | `double` | `1e-5` | Minimum volume threshold for arrangement cell filtering. Cells below this volume will be merged into adjacent cells. |
| `face_count_threshold` | `size_t` | `200` | Minimum face count threshold for arrangement cell filtering. Cells below this face count will be merged into adjacent cells. |

### Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cyclic` | `bool` | `false` | Whether the trajectory is cyclic. ⚠️ This feature is experimental and not fully supported. |

## Configuration Examples

The following examples show how to configure both grid parameters and sweep options together:

<details>
<summary><b>C++ Example</b></summary>

```c++
#include <sweep/generalized_sweep.h>

// Configure grid
sweep::GridSpec grid;
grid.resolution = {8, 8, 8};
grid.bbox_min = {-0.5, -0.5, -0.5};
grid.bbox_max = {1.5, 1.5, 1.5};

// Configure sweep options
sweep::SweepOptions options;
options.epsilon_env = 1e-3;
options.epsilon_sil = 1e-3;
options.with_insideness_check = false;
options.max_split = 1000000;

// Compute sweep
auto result = sweep::generalized_sweep(f, grid, options);
```

</details>

<details>
<summary><b>Python Example</b></summary>

```python
import sweep3d

# Configure grid
grid_spec = sweep3d.GridSpec()
grid_spec.resolution = [8, 8, 8]
grid_spec.bbox_min = [-0.5, -0.5, -0.5]
grid_spec.bbox_max = [1.5, 1.5, 1.5]

# Configure sweep options
options = sweep3d.SweepOptions()
options.epsilon_env = 1e-3
options.epsilon_sil = 1e-3
options.with_insideness_check = False
options.max_split = 1000000

# Compute sweep
result = sweep3d.generalized_sweep(my_function, grid_spec, options)
```

</details>

<details>
<summary><b>YAML Example</b></summary>

```yaml
grid:
    resolution: [8, 8, 8]
    bbox_min: [-0.5, -0.5, -0.5]
    bbox_max: [1.5, 1.5, 1.5]

parameters:
    epsilon_env: 1e-3
    epsilon_sil: 1e-3
    with_insideness_check: false
    max_split: 1000000
```

</details>
