#!/bin/bash

sbatch --account=aip-florian7 \
    --gres=gpu:l40s:1 \
    --mem=10G \
    --time=2:00:00 \
    --cpus-per-task=8 \
    --ntasks=1 \
    ./script_launchpoint/launch_attack.sh imagenet script_logpoint/weaker_priorno_increments 50 0 fake attack 1.0

sbatch --account=aip-florian7 \
    --gres=gpu:l40s:1 \
    --mem=10G \
    --time=2:00:00 \
    --cpus-per-task=8 \
    --ntasks=1 \
    ./script_launchpoint/launch_attack.sh imagenet script_logpoint/weaker_priorewc 50 1 fake attack 1.0

sbatch --account=aip-florian7 \
    --gres=gpu:l40s:1 \
    --mem=10G \
    --time=2:00:00 \
    --cpus-per-task=8 \
    --ntasks=1 \
    ./script_launchpoint/launch_attack.sh imagenet script_logpoint/weaker_priorno_ewc_mse_anchor 50 0 mse attack 1.0

sbatch --account=aip-florian7 \
    --gres=gpu:l40s:1 \
    --mem=10G \
    --time=2:00:00 \
    --cpus-per-task=8 \
    --ntasks=1 \
    ./script_launchpoint/launch_attack.sh imagenet script_logpoint/weaker_priorno_ewc_spectral_anchor_attack 50 0 spectral attack 1.0

sbatch --account=aip-florian7 \
    --gres=gpu:l40s:1 \
    --mem=10G \
    --time=2:00:00 \
    --cpus-per-task=8 \
    --ntasks=1 \
    ./script_launchpoint/launch_attack.sh imagenet script_logpoint/weaker_priorno_ewc_spectral_anchor_defense 50 0 spectral defense 1.0

sbatch --account=aip-florian7 \
    --gres=gpu:l40s:1 \
    --mem=10G \
    --time=2:00:00 \
    --cpus-per-task=8 \
    --ntasks=1 \
    ./script_launchpoint/launch_attack.sh imagenet script_logpoint/weaker_priorno_ewc_spectral_anchor_attack_smaller_weight 50 0 spectral attack 0.01

sbatch --account=aip-florian7 \
    --gres=gpu:l40s:1 \
    --mem=10G \
    --time=2:00:00 \
    --cpus-per-task=8 \
    --ntasks=1 \
    ./script_launchpoint/launch_attack.sh imagenet script_logpoint/weaker_priorno_ewc_spectral_anchor_defense_smaller_weight 50 0 spectral defense 0.01


