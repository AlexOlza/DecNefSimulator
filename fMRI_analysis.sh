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
DEVICE="cuda:1"

target_classes=(0 4)

# Each entry corresponds to the alternative classes for the matching target_classes entry
non_target_classes_0=(1 3 7)
non_target_classes_1=(5 9 1)

# ------------------------------------------------------------------
# Analyze trajectories
# ------------------------------------------------------------------

for EXP_NAME in "$fMRI_EXP"; do

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
	    python3 analyze_fMRI_trajs.py "$fMRI_EXP" "$target_class_idx" "$non_target_class_idx" 8 256 4 "$REDUCTION" "$DEVICE"
	    
	    
        done
    done
done

cd ../run_exp
python3 baseline_generators.py "$EXP_NAME" \
	--trajectory_random_seed_init "$trajectory_random_seed_init" \
	--n_trajectories "$n_trajectories" \
	--target_class_idx "$target_class_idx" \
	--non_target_class_idx "$non_target_class_idx" \
	--lambda_inv "$lambda_inv" \
	--gamma_inv "$gamma_inv" \
	--decnef_iters "$decnef_iters" \
	--ignore_classifier "$ignore_classifier" \
	--update_rule_idx "$update_rule_idx" \
	--dataset "$DATASET" \
	--device "$DEVICE" \
	--subject "$SUBJECT" \
	--z_dim "$ZDIM"
	    
