#!/usr/bin/env python3
"""Run generalized_sweep with specified config and output logging."""

import shutil
import subprocess
import sys
from pathlib import Path

# Configuration
EXECUTABLE = "./build/generalized_sweep"
# CONFIG_FILE = "./data/csg/config_cube_fine.yaml"
CONFIG_FILE = "./data/csg/config_general_set_small_test.yaml"  
CONFIG_FILE = "./data/csg/config_tet3d_set_test.yaml"

# CONFIG_FILE = "./data/csg/config_uniform_small_box.yaml"  
# CONFIG_FILE = "./data/csg/config_cube_fine.yaml"  
# CONFIG_FILE = "./data/csg/config_cube_circling.yaml"  
# CONFIG_FILE = "./data/csg/config_tet3d_deform_sub.yaml"
# INPUT_FILE = "./data/csg/mobius_cube_pert2_rotate4.yaml"
# INPUT_FILE = "./data/csg/tet3d_rotate_90_order.yaml"
# INPUT_FILE = "./data/csg/tet_deform_subtree.yaml"
# INPUT_FILE = "./data/csg/tet3d_180_origin.yaml"
INPUT_FILE = "./data/csg/tet3d_diff.yaml"
# INPUT_FILE = "./data/csg/tet3d_rotate_90_trans.yaml"
# INPUT_FILE = "./data/csg/tet3d_180_pert_rotate_center_trans.yaml"
# INPUT_FILE = "./data/csg/cube_sphere_deform_180.yaml"
# INPUT_FILE = "./data/csg/cut_torus_stroke.yaml"
# INPUT_FILE = "./data/csg/mobius_cube_test_root_rotate2.yaml"
# INPUT_FILE = "./data/csg/tet3d_180_pert_large.yaml"
# OUTPUT_DIR = "./output/mocube_visual_0.0001_0.0001_0.001_pert2_rotate4"
# OUTPUT_DIR = "./output/tet3d_deform_0.0004_0.0004_0.02_deform_fine_test_rootTrans"
# OUTPUT_DIR = "./output/tet3d_rotate_90_trans_small_box_test_mark_tet_bin"
OUTPUT_DIR = "./output/tet3d_diff_3dom_0.0002_0ring_separator"
LOG_FILE = str(Path(OUTPUT_DIR) / "running_log.txt")

# 1.782033 0.468799 0.539532
# 1.812822 0.468740 0.609160
# 1.874866 0.375152 0.656175
# 1.812391 0.559389 0.516760
# 1.781621 0.468727 0.539104
# 1.812619 0.468586 0.609081
# 1.874855 0.468660 0.562561

def main():
    # Make sure the output directory exists
    # Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy input + config into the output dir for reproducibility
    shutil.copy2(CONFIG_FILE, out_dir / Path(CONFIG_FILE).name)
    shutil.copy2(INPUT_FILE, out_dir / Path(INPUT_FILE).name)


    cmd = [
        EXECUTABLE,
        "-c", CONFIG_FILE,
        "-f", INPUT_FILE,
        OUTPUT_DIR,
    ]

    print(f"Running: {' '.join(cmd)}")
    print(f"Logging to: {LOG_FILE}")

    # Open log file and stream stdout+stderr into it
    with open(LOG_FILE, "w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)

    if result.returncode == 0:
        print("Done.")
    else:
        print(f"Command failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()