#!/usr/bin/env bash
export NCCL_P2P_LEVEL=SYS
export NCCL_DEBUG=INFO
export OMNI_USER_DIRECTORY="/home/rriemer/IsaacLab"

VIDEO_DEST="/home/rriemer/Desktop/R1-Training-RR-1sept"
mkdir -p "$VIDEO_DEST"
cd /home/rriemer/IsaacLab

echo "🚀 Launching 8-GPU Distributed Training with Head Stabilization from model_400.pt..."

/home/rriemer_google_com/env_isaacsim/bin/torchrun --nnodes=1 --nproc_per_node=8     scripts/reinforcement_learning/rsl_rl/train.py     --task Isaac-Velocity-Flat-R1-v0     --num_envs 8192     --headless     --distributed     --resume     --load_run "2026-09-01_13-10-33"     --checkpoint "model_400.pt"     --max_iterations 25000     --video     --video_length 250     --video_interval 2500     --enable_cameras 2>&1 | tee -a /home/rriemer/IsaacLab/training_8gpu_distributed.log
