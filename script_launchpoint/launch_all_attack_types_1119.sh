#!/bin/bash

source /home/bransthr/.envs/csc2529/bin/activate

# Launch without increment
python test_build_consecutive_naive_dip.py \
    --target_img_domain imagenet \
    --save_dir script_logpoint/no_increments \
    --attack_length 50 \
    --anchoring_loss_fn fake

# Launch with EWC (shouldn't work)
python test_build_consecutive_naive_dip.py \
    --target_img_domain imagenet \
    --save_dir script_logpoint/no_increments \
    --attack_length 50 \
    --anchoring_loss_fn fake \
    --update_ewc True

# Launch without EWC, with MSE anchoring to attack
python test_build_consecutive_naive_dip.py \
    --target_img_domain imagenet \
    --save_dir script_logpoint/no_increments \
    --attack_length 50 \
    --anchoring_loss_fn mse

# Launch without EWC, with spectral anchoring to defense
python test_build_consecutive_naive_dip.py \
    --target_img_domain imagenet \
    --save_dir script_logpoint/no_increments \
    --attack_length 50 \
    --anchoring_loss_fn spectral \
    --anchor_to defense

# Launch without EWC, with spectral anchoring to attack
python test_build_consecutive_naive_dip.py \
    --target_img_domain imagenet \
    --save_dir script_logpoint/no_increments \
    --attack_length 50 \
    --anchoring_loss_fn spectral \
    --anchor_to attack