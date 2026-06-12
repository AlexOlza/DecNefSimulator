#!/usr/bin/env bash

FASHION_EXP="$1"
fMRI_EXP="$2"
REDUCTION="$3"
trajectory_random_seed_init=42
n_trajectories=100
lambda_inv=4
gamma_inv=4
decnef_iters=500
ignore_classifier=0
update_rule_idx=1
DEVICE="cuda:0"
INITIAL_SEED=42
UR=1

target_classes=(0 4)

# Each entry corresponds to the alternative classes for the matching target_classes entry
non_target_classes_0=(1 3 7)
non_target_classes_1=(5 9 1)

# ------------------------------------------------------------------
# Analyze trajectories
# ------------------------------------------------------------------

for EXP_NAME in "$FASHION_EXP"; do

    if [ "$EXP_NAME" = "$FASHION_EXP" ]; then
        DATASET="FASHION"
        SUBJECT=0
        ZDIM=2
    else
        DATASET="synth_fMRI_FASHION"
        SUBJECT=8
        ZDIM=256
    fi

    for i in "${!target_classes[@]}"; do

        target_class_idx=${target_classes[$i]}

        if [ "$i" -eq 0 ]; then
            NON_TARGETS=("${non_target_classes_0[@]}")
        else
            NON_TARGETS=("${non_target_classes_1[@]}")
        fi

        for non_target_class_idx in "${NON_TARGETS[@]}"; do

            python3 analyze_trajectories.py "$EXP_NAME" "$target_class_idx" "$non_target_class_idx" 0 2 4 "$REDUCTION" "$DEVICE"
        done
    done
done
cd ../run_exp
            ./traditional_decnef_simulation.sh $EXP_NAME $INITIAL_SEED $NTRRAJS $tgt_class_idx $non_tgt_class_idx $L_inv $L_inv $NITER $UR $IGDIS $DATASET $DEVICE $SUBJECT
