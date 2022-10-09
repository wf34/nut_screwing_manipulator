#!/usr/bin/env bash
set -x

export EXAMPLES_LOCAL_DRAKE_PATH=/home/wf34/projects/drake
export BAZEL_CXXOPTS="-std=c++17"
bazel run //nut_screwing:run_manipulator -- --controller_type differential
#bazel run //nut_screwing:teleop_manipulator -- -w
