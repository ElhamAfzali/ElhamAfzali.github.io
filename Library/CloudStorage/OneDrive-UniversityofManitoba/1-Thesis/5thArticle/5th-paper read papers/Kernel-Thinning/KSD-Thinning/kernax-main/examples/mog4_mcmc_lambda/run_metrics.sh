#!/bin/bash

set -e
set -x

ALGORITHMS=("nuts" "mala")
STEP_SIZES=(0.05 0.1 0.5) 
THINNING_SIZES=(100 300 500 700 900 1100)
export LAMBDA_ENTROPY="inverse"
export BASEDIR="$( cd "$( dirname "$0" )" && pwd )"
export BASEDIR="${BASEDIR}/lambda_inverse"

for ALGO in ${ALGORITHMS[@]}; do
    export ALGORITHM=$ALGO
    for THINNING_SIZE in ${THINNING_SIZES[@]}; do
        export THIN_SIZE=$THINNING_SIZE
        for val in ${STEP_SIZES[@]}; do
            export STEP_SIZE=$val
            python compute_mmd.py --algorithm=$ALGORITHM --step_size=$STEP_SIZE --thinning_size=$THIN_SIZE --srcdir=$BASEDIR
        done
    done
done

export LAMBDA_ENTROPY="sq_inverse"
export BASEDIR="$( cd "$( dirname "$0" )" && pwd )"
export BASEDIR="${BASEDIR}/lambda_sq_inverse"

for ALGO in ${ALGORITHMS[@]}; do
    export ALGORITHM=$ALGO
    for THINNING_SIZE in ${THINNING_SIZES[@]}; do
        export THIN_SIZE=$THINNING_SIZE
        for val in ${STEP_SIZES[@]}; do
            export STEP_SIZE=$val
            python compute_mmd.py --algorithm=$ALGORITHM --step_size=$STEP_SIZE --thinning_size=$THIN_SIZE --srcdir=$BASEDIR
        done
    done
done

export LAMBDA_ENTROPY="log_inverse"
export BASEDIR="$( cd "$( dirname "$0" )" && pwd )"
export BASEDIR="${BASEDIR}/lambda_log_inverse"

for ALGO in ${ALGORITHMS[@]}; do
    export ALGORITHM=$ALGO
    for THINNING_SIZE in ${THINNING_SIZES[@]}; do
        export THIN_SIZE=$THINNING_SIZE
        for val in ${STEP_SIZES[@]}; do
            export STEP_SIZE=$val
            python compute_mmd.py --algorithm=$ALGORITHM --step_size=$STEP_SIZE --thinning_size=$THIN_SIZE --srcdir=$BASEDIR
        done
    done
done