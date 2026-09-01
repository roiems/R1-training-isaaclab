# Unitree R1 Humanoid Bipedal Locomotion in NVIDIA Isaac Lab

[![Isaac Sim](https://img.shields.io/badge/NVIDIA%20Isaac%20Sim-6.0-green.svg)](https://docs.isaacsim.omniverse.nvidia.com/)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-6.1-blue.svg)](https://isaac-sim.github.io/IsaacLab/)
[![RSL--RL](https://img.shields.io/badge/RSL--RL-2.3%2B-orange.svg)](https://github.com/leggedrobotics/rsl_rl)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-red.svg)](https://pytorch.org/)

High-performance reinforcement learning training configurations, environment definitions, pre-trained policy checkpoints, and real-time interactive teleoperation scripts for the **Unitree R1 Humanoid Robot** in **NVIDIA Isaac Sim 6.0** and **Isaac Lab 6.1**.

---

## Features

- **Full-Body 26-DOF Control:** Coordinates legs (hip pitch/roll/yaw, knee, ankle pitch/roll), dual-axis waist, 3-axis arms/shoulders, elbows, wrists, and head.
- **Natural Bipedal Gait:** Engineered reward functions with calibrated foot airtime (`feet_air_time_positive_biped`), upright pelvis posture stabilization, and anti-foot-slip penalties.
- **Zero-Velocity Stable Standing:** Clean transition between dynamic walking and zero-velocity idle standing without collapsing.
- **Multi-Perspective Dynamic Camera Tracking:** Real-time camera that dynamically follows the robot orientation with hotkeys for **Third-Person Chase Cam**, **First-Person POV (Head Cam)**, **Over-the-Shoulder**, and **Isometric**.
- **Photorealistic Environment Support:** Ready to run in standard flat planes or complex USD environments (such as NVIDIA Simple Warehouse).
- **Multi-GPU Scalability:** Distributed training pipelines tested with 8x NVIDIA GPUs running 65,536 concurrent simulation environments.
- **Production-Ready Export:** Pre-exported PyTorch (`.pt`), TorchScript JIT (`.jit`), and ONNX (`.onnx`) models for edge and physical robot deployment.

---

## Repository Structure

```text
R1-training-isaaclab/
├── README.md                          # Project documentation and usage guide
├── requirements.txt                   # Dependencies and package requirements
├── configs/
│   ├── env/
│   │   ├── flat_env_cfg.py            # Flat plane MDP task configuration
│   │   ├── rough_env_cfg.py           # Procedural rough terrain MDP task configuration
│   │   └── env.yaml                   # Exact environment parameters dump
│   ├── agents/
│   │   ├── rsl_rl_ppo_cfg.py          # PPO actor-critic network & hyperparameters
│   │   └── agent.yaml                 # Exact agent parameters dump
│   └── robots/
│       └── unitree_r1.py              # R1 robot articulation, limits, and PD gains
├── models/
│   ├── checkpoints/
│   │   ├── model_600.pt               # Base trained policy checkpoint
│   │   ├── model_1000.pt              # Step 1000 checkpoint
│   │   └── model_1800.pt              # Advanced step 1800 checkpoint
│   └── exported/
│       ├── policy.pt                  # Clean PyTorch policy weights
│       ├── policy.jit                 # Standalone TorchScript JIT policy
│       ├── policy.onnx                # Standalone ONNX runtime policy
│       └── policy.onnx.data           # ONNX tensor weights
├── robot_description/
│   ├── R1.urdf                        # Unitree R1 kinematics and joint URDF
│   ├── meshes/                        # 3D STL collision and visual meshes
│   └── R1/                            # Converted Isaac Sim USD assets and textures
└── scripts/
    ├── play_r1_teleop.py              # Interactive teleoperation & camera player
    ├── run_teleop.sh                  # One-click launch script for teleoperation
    ├── train_r1_single_gpu.sh         # Single-GPU training launcher
    └── train_r1_distributed_8gpu.sh   # 8-GPU distributed training launcher (torchrun)
```

---

## Quick Start: Run Pre-Trained Policy

### 1. Environment Setup
Make sure NVIDIA Isaac Sim 6.0 and Isaac Lab are installed and the virtual environment is activated:

```bash
source ~/env_isaacsim/bin/activate
cd ~/IsaacLab
```

### 2. Interactive WASD Teleoperation (Flat Ground)
```bash
./isaaclab.sh -p /path/to/R1-training-isaaclab/scripts/play_r1_teleop.py \
    --task Isaac-Velocity-Flat-R1-v0 \
    --num_envs 1 \
    --checkpoint /path/to/R1-training-isaaclab/models/checkpoints/model_600.pt \
    --viz kit
```

### 3. Interactive Teleoperation in NVIDIA Simple Warehouse
```bash
./isaaclab.sh -p /path/to/R1-training-isaaclab/scripts/play_r1_teleop.py \
    --task Isaac-Velocity-Flat-R1-v0 \
    --num_envs 1 \
    --checkpoint /path/to/R1-training-isaaclab/models/checkpoints/model_600.pt \
    --viz kit \
    --warehouse https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/Environments/Simple_Warehouse/warehouse.usd
```

---

## Interactive Controls & Camera Modes

Click inside the 3D viewport window to give it keyboard focus:

| Key | Action | Description |
| :---: | :--- | :--- |
| **`W`** | **Walk Forward** | Linear velocity $v_x > 0$ |
| **`S`** | **Walk Backward** | Linear velocity $v_x < 0$ |
| **`A`** | **Turn Left** | Angular velocity $\omega_z > 0$ |
| **`D`** | **Turn Right** | Angular velocity $\omega_z < 0$ |
| **`Q` / `E`** | **Strafe Left / Right** | Linear velocity $v_y$ |
| **`L`** | **Zero / Stand** | Instantly resets all velocity commands to zero |
| **`C` / `V`** | **Cycle Camera View** | Toggles between Behind Chase, Head POV, Shoulder, and Isometric |

### Camera Modes:
1. **Third-Person Chase (Default):** Camera sits 2.5m behind the robot and rotates with the robot heading.
2. **First-Person POV:** Camera sits directly inside the robot head (eye-level), showing the robot forward field of view.
3. **Over-The-Shoulder:** Cinematic third-person perspective over the right shoulder.
4. **Isometric Overview:** High diagonal perspective for broad warehouse navigation.

---

## Training the Policy

### Architecture & MDP Specification
- **Observation Space (90-dim):**
  - Base Linear Velocity (`base_lin_vel`, 3-dim)
  - Base Angular Velocity (`base_ang_vel`, 3-dim)
  - Projected Gravity Vector (`projected_gravity`, 3-dim)
  - Velocity Commands (`velocity_commands`, 3-dim)
  - Joint Positions (`joint_pos`, 26-dim)
  - Joint Velocities (`joint_vel`, 26-dim)
  - Previous Actions (`actions`, 26-dim)
- **Action Space (26-dim):** Target joint position offsets for all 26 controllable joints.
- **Algorithm:** PPO with Generalized Advantage Estimation (GAE), ELU activations, and MLP Actor-Critic `[512, 256, 128]`.

### Single-GPU Training
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-R1-v0 \
    --num_envs 4096 \
    --headless \
    --max_iterations 5000
```

### Multi-GPU Distributed Training (8x GPUs)
```bash
torchrun --nnodes=1 --nproc_per_node=8 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-R1-v0 \
    --num_envs 8192 \
    --headless \
    --distributed \
    --max_iterations 25000
```

---

## Exported Policy Deployment

The trained policy can be evaluated or deployed using the standalone exports in `models/exported/`:

- **PyTorch:** `models/exported/policy.pt`
- **TorchScript JIT:** `models/exported/policy.jit` (can be executed in C++ or Python without Isaac Lab dependencies)
- **ONNX:** `models/exported/policy.onnx` (for TensorRT, ONNX Runtime, or edge robotics controllers)

```python
import torch

# Load JIT policy
policy = torch.jit.load("models/exported/policy.jit")
policy.eval()

# Run inference with a 90-dimensional observation tensor
obs = torch.zeros(1, 90)
actions = policy(obs)
print("Computed joint position targets:", actions)
```
