#!/bin/bash

set -e
mkdir -p mcmc_data

ALGORITHMS=("nuts" "mala")
STEP_SIZES=(0.05 0.1 0.5) 
THINNING_SIZES=(100 300 500 700 900 1100)
export BASEDIR="$( cd "$( dirname "$0" )" && pwd )"
for ALGO in ${ALGORITHMS[@]}; do
    export ALGORITHM=$ALGO
    for THINNING_SIZE in ${THINNING_SIZES[@]}; do
        export THIN_SIZE=$THINNING_SIZE
        for val in ${STEP_SIZES[@]}; do
            export STEP_SIZE=$val
            python ../mog_mcmc.py --algorithm=$ALGORITHM --step_size=$STEP_SIZE --thinning_size=$THIN_SIZE --output_path=${BASEDIR}/mcmc_data
        done
    done
done