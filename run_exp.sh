#!/usr/bin/env bash

FASHION_EXP="$1"
fMRI_EXP="$2"
CLF="$3" # 0 for DecNef simulations, 1 for control simulations with random feedback
trajectory_random_seed_init=42
NTRAJS=100
L_inv=4
NITER=500
GENERATOR_EPOCHS=25
UR=1
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
cd src/DecNefSimulator/run_exp
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

        TGT_CLASS=${target_classes[$i]}

        if [ "$i" -eq 0 ]; then
            NON_TARGETS=("${non_target_classes_0[@]}")
        else
            NON_TARGETS=("${non_target_classes_1[@]}")
        fi

        for ALT_CLASS in "${NON_TARGETS[@]}"; do
            SECONDS=0
		python3 traditional_decnef_n_instances.py  $EXP_NAME \
		    --trajectory_random_seed_init $INITIAL_SEED \
		    --n_trajectories $NTRAJS \
		    --target_class_idx $TGT_CLASS \
		    --non_target_class_idx $ALT_CLASS \
		    --lambda_inv $L_inv \
		    --gamma_inv $L_inv \
		    --decnef_iters $NITER \
		    --ignore_classifier $CLF \
		    --update_rule_idx $UR \
		    --dataset $DATASET \
		    --device $DEVICE \
		    --subject $SUBJECT \
		    --z_dim $ZDIM \
		    --generator_epochs $GENERATOR_EPOCHS
		DECNEF_duration=$SECONDS
		echo "DecNef ($((n_trajectories))): $((DECNEF_duration / 60)) minutes and $((DECNEF_duration % 60)) seconds."

		total_duration=$(($CNN_duration+$VAE_duration+$DECNEF_duration))
		echo "Total time: $((total_duration / 60)) minutes and $((total_duration % 60)) seconds."
        done
    done
done
            
