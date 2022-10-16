#!/usr/bin/env bash
set -x

export EXAMPLES_LOCAL_DRAKE_PATH=/home/wf34/projects/drake
export BAZEL_CXXOPTS="-std=c++17"
bazel run //nut_screwing:do_graphing -- \
    --input_telemetry /home/wf34/projects/control_nut_screwing_manipulator/m.log \
    --output_graph /home/wf34/projects/control_nut_screwing_manipulator/m.png
