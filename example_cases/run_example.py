#!/usr/bin/env python3
"""Run generalized_sweep with specified config and output logging."""

import shutil
import subprocess
import sys
from pathlib import Path

example_cases = ["cad_bolt",  #fig 17 
                 "cube_circling_rotate", #fig 14
                 "cube_circling_rotate_expand", #fig 16
                 "high_genus_cut1",  #fig 19 sweep 1
                 "high_genus_cut2",  #fig 19 sweep 2
                 "tet_torus_cut",  #fig 18
                 "tet3d_diff_R60", #fig 1
                 "tet3d_rotate",   #fig 2&7
                 "tet3d_rotate_deform" #fig 15
                 ] 
#"cad_bolt" case is memory demanding, to run the case with paper setting, you need 256 RAM cpu to run
# If your pc memory is limited, you may use larger epsilon_env, epsilon_sil values, and min_tet_edge_length
#change the example case to the folder name in the example_cases folder
example_case = "tet3d_diff_R60"
# Configuration
EXECUTABLE = "../build/generalized_sweep"
CONFIG_FILE = example_case + "/config.yaml"  
INPUT_FILE = example_case + "/" + example_case + ".yaml"
OUTPUT_DIR = "./output/" + example_case
LOG_FILE = str(Path(OUTPUT_DIR) / "running_log.txt")


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
        # "--static",
        # "--st", " 0" ,
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