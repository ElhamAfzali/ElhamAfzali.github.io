#!/bin/bash

set -e
mkdir -p mcmc_data

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
            python ../mog_mcmc.py --algorithm=$ALGORITHM --step_size=$STEP_SIZE --output_path=${BASEDIR}/mcmc_data --dim=$DIMENSION --case "mog4" --num_iterations=100000 --thinning_size=300
        done
    done
done