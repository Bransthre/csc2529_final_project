#!/bin/bash
#SBATCH --output=script_logpoint/slurm_files/R_%j.out
#SBATCH --error=script_logpoint/slurm_files/R_%j.err
#SBATCH --account=aip-florian7

source /home/bransthr/.envs/csc2529/bin/activate

export SAVE_DIR=$1
export UPDATE_EWC=$2
export ANCHORING_LOSS_FN=$3
export MASK_ALPHA=$4
export ANCHORING_LOSS_WEIGHT=$5
export RETRAIN_DIP_EVERY_ATTACK=$6
export ANCHOR_WITH_PAST_EXAMPLES=$7
export NUM_ITER_PER_IMAGE=$8
export SEQUENCE_SEED=$9

python run_experiments.py \
    --target_img_domain imagenet \
    --attack_length 50 \
    --save_dir $SAVE_DIR \
    --update_ewc $UPDATE_EWC \
    --anchoring_loss_fn $ANCHORING_LOSS_FN \
    --fourier_mask_alpha $MASK_ALPHA \
    --anchoring_loss_weight $ANCHORING_LOSS_WEIGHT \
    --retrain_dip_every_attack $RETRAIN_DIP_EVERY_ATTACK \
    --anchor_with_past_examples $ANCHOR_WITH_PAST_EXAMPLES \
    --num_iter_per_image $NUM_ITER_PER_IMAGE \
    --sequence_seed $SEQUENCE_SEED