#!/bin/bash
#SBATCH --output=script_logpoint/slurm_files/R_%j.out
#SBATCH --error=script_logpoint/slurm_files/R_%j.err
#SBATCH --account=aip-florian7

source /home/bransthr/.envs/csc2529/bin/activate

export TARGET_IMG_DOMAIN=$1
export SAVE_DIR=$2
export ATTACK_LENGTH=$3
export UPDATE_EWC=$4
export ANCHORING_LOSS_FN=$5
export ANCHOR_TO=$6
export ANCHORING_LOSS_WEIGHT=$7

python test_build_consecutive_naive_dip.py \
    --target_img_domain $TARGET_IMG_DOMAIN \
    --save_dir $SAVE_DIR \
    --attack_length $ATTACK_LENGTH \
    --update_ewc $UPDATE_EWC \
    --anchoring_loss_fn $ANCHORING_LOSS_FN \
    --anchor_to $ANCHOR_TO \
    --anchoring_loss_weight $ANCHORING_LOSS_WEIGHT
