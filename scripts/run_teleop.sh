#!/usr/bin/env bash
export DISPLAY=:10
export XAUTHORITY=/var/opt/thinlinc/sessions/rriemer/10/Xauthority
export LD_PRELOAD=/home/rriemer_google_com/env_isaacsim/lib/python3.12/site-packages/isaacsim/extscache/omni.gpu_foundation-0.0.0+6312fa25.lx64.r.cp312/bin/deps/libglib-2.0.so.0

source /home/rriemer/env_isaacsim/bin/activate
cd /home/rriemer/IsaacLab

./isaaclab.sh -p /home/rriemer/cosmos_workspace/play_r1_teleop.py     --task Isaac-Velocity-Flat-R1-v0     --num_envs 1     --checkpoint /home/rriemer/IsaacLab/logs/rsl_rl/r1_flat_locomotion/2026-09-01_13-50-10/model_2200.pt     --viz kit     --warehouse https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/Environments/Simple_Warehouse/warehouse.usd     "$@"
