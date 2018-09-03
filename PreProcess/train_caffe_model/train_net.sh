#!/usr/bin/env sh
set -e

~/caffe/build/tools/caffe train --solver=RRNwithQP_solver.prototxt --gpu all 2>&1 |tee /home/zsf/coding/caffe-net/I_RRN/log/out.log



