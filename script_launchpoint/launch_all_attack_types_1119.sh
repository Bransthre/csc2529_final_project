#!/bin/bash

mask_alphas=(1 5 10)
anchoring_weight_mse=(0.5 1 2)
anchoring_weight_spectral=(0.0005 0.001 0.002)
num_iter=(1200 1800 2400)

seeds=(1 2 3 4)

for seed in "${seeds[@]}"; do
    # REGION A: Baselines
    sbatch --account=aip-florian7 \
        --gres=gpu:l40s:1 \
        --mem=35G \
        --time=12:00:00 \
        --cpus-per-task=8 \
        --ntasks=1 \
        ./script_launchpoint/launch_attack.sh script_logpoint_1123/incremental/vanilla_${seed} 0 fake 1.0 1.0 0 1 2400 $seed

    sbatch --account=aip-florian7 \
        --gres=gpu:l40s:1 \
        --mem=35G \
        --time=20:00:00 \
        --cpus-per-task=8 \
        --ntasks=1 \
        ./script_launchpoint/launch_attack.sh script_logpoint_1123/incremental/ewc_${seed} 1 fake 1.0 1.0 0 1 2400 $seed

    for num_iter_val in "${num_iter[@]}"; do
        sbatch --account=aip-florian7 \
            --gres=gpu:l40s:1 \
            --mem=35G \
            --time=12:00:00 \
            --cpus-per-task=8 \
            --ntasks=1 \
            ./script_launchpoint/launch_attack.sh script_logpoint_1123/retrain/use_anchor_ni${num_iter_val}_${seed} 0 mse 1.0 1.0 1 1 $num_iter_val $seed

        sbatch --account=aip-florian7 \
            --gres=gpu:l40s:1 \
            --mem=35G \
            --time=12:00:00 \
            --cpus-per-task=8 \
            --ntasks=1 \
            ./script_launchpoint/launch_attack.sh script_logpoint_1123/retrain/individual_ni${num_iter_val}_${seed} 0 mse 1.0 1.0 1 0 $num_iter_val $seed

        # REGION B: MSE Anchoring
        for anchor_weight in "${anchoring_weight_mse[@]}"; do
            sbatch --account=aip-florian7 \
                --gres=gpu:l40s:1 \
                --mem=35G \
                --time=12:00:00 \
                --cpus-per-task=8 \
                --ntasks=1 \
                ./script_launchpoint/launch_attack.sh script_logpoint_1123/incremental/mse_anchor_aw${anchor_weight}_ni${num_iter_val}_${seed} 0 mse 1.0 $anchor_weight 0 1 $num_iter_val $seed
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
                    ./script_launchpoint/launch_attack.sh script_logpoint_1123/incremental/spectral_anchor_ma${mask_alpha}_aw${anchor_weight}_ni${num_iter_val}_${seed} 0 spectral $mask_alpha $anchor_weight 0 1 $num_iter_val $seed
            done
        done

    done
done
