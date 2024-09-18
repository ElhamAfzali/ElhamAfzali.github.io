#!/bin/bash

set -e
mkdir -p mmd_data

ALGORITHMS=("nuts" "mala")
STEP_SIZES=(0.05 0.1 0.5)
DIMENSIONS=(10 20 30 40 50 60 70 80 90 100)
export BASEDIR="$( cd "$( dirname "$0" )" && pwd )"
for ALGO in ${ALGORITHMS[@]}; do
    export ALGORITHM=$ALGO
    for val in ${STEP_SIZES[@]}; do
        export STEP_SIZE=$val
        for DIMS in ${DIMENSIONS[@]}; do
            export DIMENSION=$DIMS
            python compute_mmd.py --algorithm=$ALGORITHM --step_size=$STEP_SIZE --dimension=$DIMENSION
        done
    done
done