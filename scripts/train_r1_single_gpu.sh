#!/usr/bin/env bash
# ==============================================================================
# train_r1_ppo.sh
# Train Unitree R1 Humanoid Bipedal Locomotion via PPO (RSL-RL) on Blackwell GPU
# 4,096 parallel environments simulated simultaneously on GPU 0
# ==============================================================================
set -e

export CUDA_VISIBLE_DEVICES=0
cd /home/rriemer/IsaacLab

echo "=================================================================="
echo "🤖 TRAINING UNITREE R1 HUMANOID LOCOMOTION WITH PPO (ISAAC LAB)"
echo "   Environments: 4,096 parallel instances on NVIDIA Blackwell GPU 0"
echo "   Task: Isaac-Velocity-Flat-R1-v0"
echo "=================================================================="

/home/rriemer_google_com/env_isaacsim/bin/python /home/rriemer/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/train.py     --task Isaac-Velocity-Flat-R1-v0     --num_envs 4096     --headless
