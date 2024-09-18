#!/bin/bash

set -e

ALGORITHMS=("mala" "nuts")
DATASETS=("BreastCancer" "Sonar" "HabermanSurvival" "LiverDisorder" "Diabetes")
STEP_SIZES=(1e-2 1e-3 1e-4 1e-5)
THINNING_SIZES=(50 100 300)
export NUM_REPITITIONS=10
export NUM_ITERATIONS=10000
export BASEDIR="$( cd "$( dirname "$0" )" && pwd )"
for ALGO in ${ALGORITHMS[@]}; do
    export ALGORITHM=$ALGO
    for DATASET in ${DATASETS[@]}; do
        export DATASET_NAME=$DATASET
        for val in ${STEP_SIZES[@]}; do
            export STEP_SIZE=$val
            for THIN_SIZE in ${THINNING_SIZES[@]}; do
                export THINNING_SIZE=$THIN_SIZE
                python logistic_regression.py --algorithm=$ALGORITHM \
                                            --dataset_name=$DATASET_NAME \
                                            --path_to_data=$PATH_TO_DATASETS \
                                            --num_iterations=$NUM_ITERATIONS \
                                            --step_size=$STEP_SIZE \
                                            --thinning_size=$THINNING_SIZE \
                                            --num_repetitions=$NUM_REPITITIONS \
                                            --output_path=${BASEDIR}/mcmc_data
            done
        done
    done
done