#!/bin/bash

mask_alphas=(1 5 10)
anchoring_weight_mse=(0.5 1 2)
anchoring_weight_spectral=(0.0005 0.001 0.002)
num_iter=(1200 1800 2400)

# REGION A: Baselines
sbatch --account=aip-florian7 \
    --gres=gpu:l40s:1 \
    --mem=35G \
    --time=12:00:00 \
    --cpus-per-task=8 \
    --ntasks=1 \
    ./script_launchpoint/launch_attack.sh script_logpoint/incremental/vanilla 0 fake 1.0 1.0 0 1 2400

sbatch --account=aip-florian7 \
    --gres=gpu:l40s:1 \
    --mem=35G \
    --time=12:00:00 \
    --cpus-per-task=8 \
    --ntasks=1 \
    ./script_launchpoint/launch_attack.sh script_logpoint/incremental/ewc 1 fake 1.0 1.0 0 1 2400

for num_iter_val in "${num_iter[@]}"; do
    sbatch --account=aip-florian7 \
        --gres=gpu:l40s:1 \
        --mem=35G \
        --time=12:00:00 \
        --cpus-per-task=8 \
        --ntasks=1 \
        ./script_launchpoint/launch_attack.sh script_logpoint/retrain/use_anchor_ni${num_iter_val} 0 mse 1.0 1.0 1 1 $num_iter_val

    sbatch --account=aip-florian7 \
        --gres=gpu:l40s:1 \
        --mem=35G \
        --time=12:00:00 \
        --cpus-per-task=8 \
        --ntasks=1 \
        ./script_launchpoint/launch_attack.sh script_logpoint/retrain/individual_ni${num_iter_val} 0 mse 1.0 1.0 1 0 $num_iter_val

    # REGION B: MSE Anchoring
    for anchor_weight in "${anchoring_weight_mse[@]}"; do
        sbatch --account=aip-florian7 \
            --gres=gpu:l40s:1 \
            --mem=35G \
            --time=12:00:00 \
            --cpus-per-task=8 \
            --ntasks=1 \
            ./script_launchpoint/launch_attack.sh script_logpoint/incremental/mse_anchor_aw${anchor_weight}_ni${num_iter_val} 0 mse 1.0 $anchor_weight 0 1 $num_iter_val
    done

    # REGION B: Spectral Anchoring
    for mask_alpha in "${mask_alphas[@]}"; do
        for anchor_weight in "${anchoring_weight_spectral[@]}"; do
            sbatch --account=aip-florian7 \
                --gres=gpu:l40s:1 \
                --mem=35G \
                --time=12:00:00 \
                --cpus-per-task=8 \
                --ntasks=1 \
                ./script_launchpoint/launch_attack.sh script_logpoint/incremental/spectral_anchor_ma${mask_alpha}_aw${anchor_weight}_ni${num_iter_val} 0 spectral $mask_alpha $anchor_weight 0 1 $num_iter_val
        done
    done

done
