# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and teleoperate trained RSL-RL policies on Unitree R1."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

# Ensure script dir and IsaacLab rsl_rl dir are on sys.path for cli_args
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
isaaclab_rsl_dir = "/home/rriemer/IsaacLab/scripts/reinforcement_learning/rsl_rl"
if os.path.exists(isaaclab_rsl_dir) and isaaclab_rsl_dir not in sys.path:
    sys.path.insert(0, isaaclab_rsl_dir)
import time
from importlib import metadata
from packaging import version

import cv2
import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply_yaw

from isaaclab.utils.string import list_intersection, string_to_callable
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx, handle_deprecated_rsl_rl_cfg
from isaaclab.utils.seed import configure_seed

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    get_checkpoint_path,
    launch_simulation,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

import cli_args  # isort: skip

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Teleoperate a trained RSL-RL locomotion policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=True, help="Run in real-time, if possible.")
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
parser.add_argument(
    "--warehouse",
    type=str,
    default=None,
    help="Path to a static scene USD to use as the ground instead of the training terrain.",
)
parser.add_argument(
    "--camera",
    type=str,
    default="chase",
    choices=["chase", "pov", "front", "shoulder", "isometric", "free"],
    help="Camera view: chase (behind looking forward), pov (robot head POV), front (facing robot), shoulder, isometric, free.",
)
parser.add_argument(
    "--status_every", type=int, default=0, help="Print commanded vs actual base velocity every N steps (0 = off)."
)
parser.add_argument(
    "--no-follow",
    dest="follow",
    action="store_false",
    default=True,
    help="Leave the viewport camera static instead of chasing the robot.",
)
parser.add_argument("--v_x", type=float, default=1.0, help="Forward/back speed sensitivity (m/s).")
parser.add_argument("--v_y", type=float, default=0.2, help="Strafe speed sensitivity (m/s).")
parser.add_argument("--w_z", type=float, default=0.8, help="Yaw rate sensitivity (rad/s).")
parser.add_argument(
    "--no-clamp",
    dest="clamp",
    action="store_false",
    default=True,
    help="Send raw keyboard commands instead of clamping them into the range the policy was trained on.",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


def _free_teleop_keys():
    """Drop Kit editor hotkeys that collide with the teleop keys."""
    try:
        from omni.kit.hotkeys.core import get_hotkey_registry

        registry = get_hotkey_registry()
    except Exception as exc:
        print(f"[WARN] hotkey registry unavailable ({exc}); Kit editor hotkeys left intact.")
        return
    freed = 0
    for key in ("W", "A", "S", "D", "Q", "E", "L", "C", "V"):
        try:
            for hotkey in list(registry.get_all_hotkeys_for_key(key)):
                if registry.deregister_hotkey(hotkey):
                    freed += 1
        except Exception:
            continue
    print(f"[INFO] released {freed} conflicting Kit editor hotkey(s).")


def _build_keyboard(device: str):
    """Create an Se2Keyboard with WASD bindings layered on top of the defaults."""
    try:
        import omni.appwindow  # noqa: F401
    except ModuleNotFoundError:
        print("[WARN] omni.appwindow unavailable (headless?) -- keyboard teleop disabled.")
        return None

    _free_teleop_keys()

    from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg

    keyboard = Se2Keyboard(
        Se2KeyboardCfg(
            v_x_sensitivity=args_cli.v_x,
            v_y_sensitivity=args_cli.v_y,
            omega_z_sensitivity=args_cli.w_z,
            sim_device=device,
        )
    )
    keyboard._INPUT_KEY_MAPPING.update({
        "W": np.asarray([1.0, 0.0, 0.0]) * args_cli.v_x,
        "S": np.asarray([-1.0, 0.0, 0.0]) * args_cli.v_x,
        "Q": np.asarray([0.0, 1.0, 0.0]) * args_cli.v_y,
        "E": np.asarray([0.0, -1.0, 0.0]) * args_cli.v_y,
        "A": np.asarray([0.0, 0.0, 1.0]) * args_cli.w_z,
        "D": np.asarray([0.0, 0.0, -1.0]) * args_cli.w_z,
    })
    return keyboard


def _apply_teleop_overrides(env_cfg):
    """Stop the command manager from fighting the keyboard."""
    vel = env_cfg.commands.base_velocity
    vel.resampling_time_range = (1.0e9, 1.0e9)
    vel.rel_standing_envs = 0.0
    vel.heading_command = False
    vel.rel_heading_envs = 0.0
    vel.debug_vis = True
    env_cfg.episode_length_s = 1.0e9
    env_cfg.events.push_robot = None

    if args_cli.follow:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (-2.5, 0.0, 1.3)
        env_cfg.viewer.lookat = (2.5, 0.0, 0.9)


def _apply_warehouse(env_cfg, usd_path: str):
    """Replace the generated obstacle course with a static warehouse USD."""
    from isaaclab.terrains import TerrainImporterCfg

    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=usd_path,
        collision_group=-1,
        debug_vis=False,
    )
    env_cfg.scene.env_spacing = 8.0
    if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
        env_cfg.curriculum.terrain_levels = None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        _apply_teleop_overrides(env_cfg)
        if args_cli.warehouse:
            _apply_warehouse(env_cfg, args_cli.warehouse)

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        env_cfg.log_dir = os.path.dirname(resume_path)

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, True)
        runner.load(resume_path)

        policy = runner.get_inference_policy(device=env.unwrapped.device)

        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        if version.parse(installed_version) >= version.parse("4.0.0"):
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            policy_nn = None
        else:
            policy_nn = runner.alg.policy if version.parse(installed_version) >= version.parse("2.3.0") else runner.alg.actor_critic
            normalizer = getattr(policy_nn, "actor_obs_normalizer", None) or getattr(
                policy_nn, "student_obs_normalizer", None
            )
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

        # -- teleop and camera wiring ------------------------------------------
        keyboard = _build_keyboard(env.unwrapped.device)
        CAMERA_MODES = ["chase", "pov", "front", "shoulder", "isometric", "free"]
        camera_state = {"mode": args_cli.camera}
        CAMERA_CONFIGS = {
            "chase": {
                "eye_offset": torch.tensor([[-2.5, 0.0, 1.3]]),
                "lookat_offset": torch.tensor([[2.5, 0.0, 0.9]]),
                "name": "Third-Person Chase (Behind Robot looking forward)",
            },
            "pov": {
                "eye_offset": torch.tensor([[0.15, 0.0, 1.30]]),
                "lookat_offset": torch.tensor([[5.0, 0.0, 1.25]]),
                "name": "First-Person POV (Robot Eyes/Head looking forward)",
            },
            "front": {
                "eye_offset": torch.tensor([[2.5, 0.0, 1.2]]),
                "lookat_offset": torch.tensor([[0.0, 0.0, 0.9]]),
                "name": "Front View (Facing Robot Head-On)",
            },
            "shoulder": {
                "eye_offset": torch.tensor([[-1.8, -0.45, 1.4]]),
                "lookat_offset": torch.tensor([[3.0, 0.0, 1.0]]),
                "name": "Over-The-Shoulder View",
            },
            "isometric": {
                "eye_offset": torch.tensor([[-3.5, -3.5, 2.5]]),
                "lookat_offset": torch.tensor([[0.0, 0.0, 0.5]]),
                "name": "Isometric / Diagonal Overview",
            },
        }

        def _cycle_camera():
            curr_idx = CAMERA_MODES.index(camera_state["mode"])
            new_idx = (curr_idx + 1) % len(CAMERA_MODES)
            camera_state["mode"] = CAMERA_MODES[new_idx]
            mode_name = CAMERA_CONFIGS.get(camera_state["mode"], {}).get("name", "Free / Manual")
            print(f"[CAMERA] Switched to: {mode_name} (Mode: {camera_state["mode"]})")

        if keyboard is not None:
            keyboard.add_callback("C", _cycle_camera)
            keyboard.add_callback("V", _cycle_camera)

        command_term = env.unwrapped.command_manager.get_term("base_velocity")
        if keyboard is not None:
            print("=" * 62)
            print(" UNITREE R1 TELEOP")
            print("   W / S : walk forward / backward")
            print("   A / D : turn left / right")
            print("   Q / E : strafe left / right")
            print("   L     : zero all commands (stand)")
            print("   C / V : cycle camera views (Behind / POV / Front / Shoulder / Iso)")
            print("=" * 62)
        else:
            command_term.vel_command_b[:] = torch.tensor(
                [0.5, 0.0, 0.0], device=env.unwrapped.device
            )

        dt = env.unwrapped.step_dt
        obs = env.get_observations()
        robot = env.unwrapped.scene["robot"]
        step = 0

        ranges = env.unwrapped.cfg.commands.base_velocity.ranges
        cmd_low = torch.tensor(
            [ranges.lin_vel_x[0], ranges.lin_vel_y[0], ranges.ang_vel_z[0]],
            device=env.unwrapped.device,
        )
        cmd_high = torch.tensor(
            [ranges.lin_vel_x[1], ranges.lin_vel_y[1], ranges.ang_vel_z[1]],
            device=env.unwrapped.device,
        )
        if args_cli.clamp:
            print(f"[INFO] clamping commands to trained range: low={cmd_low.tolist()} high={cmd_high.tolist()}")
        try:
            while True:
                start_time = time.time()
                with torch.inference_mode():
                    if keyboard is not None:
                        cmd = keyboard.advance()
                        if args_cli.clamp:
                            if torch.norm(cmd) > 1e-4:
                                cmd = torch.clamp(cmd, cmd_low, cmd_high)
                        command_term.vel_command_b[:] = cmd
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)

                    # --- Dynamic Camera Tracking in Direction Robot Faces ---
                    cur_mode = camera_state["mode"]
                    if cur_mode in CAMERA_CONFIGS and args_cli.follow:
                        cfg = CAMERA_CONFIGS[cur_mode]
                        p_root = robot.data.root_pos_w[0:1]
                        q_root = robot.data.root_quat_w[0:1]
                        dev = p_root.device
                        e_off = cfg["eye_offset"].to(dev)
                        l_off = cfg["lookat_offset"].to(dev)
                        if cur_mode == "isometric":
                            c_eye = (p_root + e_off)[0].tolist()
                            c_look = (p_root + l_off)[0].tolist()
                        else:
                            c_eye = (p_root + quat_apply_yaw(q_root, e_off))[0].tolist()
                            c_look = (p_root + quat_apply_yaw(q_root, l_off))[0].tolist()
                        env.unwrapped.sim.set_camera_view(eye=c_eye, target=c_look)
                    
                    # --- Stream Live Frame to Cosmos Studio ---
                    try:
                        frame = env.render()
                        if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                            sim_out_dir = "/home/rriemer/cosmos_workspace/sim_output"
                            os.makedirs(sim_out_dir, exist_ok=True)
                            tmp_p = os.path.join(sim_out_dir, "live_sim_frame.tmp.png")
                            dst_p = os.path.join(sim_out_dir, "live_sim_frame.png")
                            bgr_f = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(tmp_p, bgr_f)
                            os.replace(tmp_p, dst_p)
                    except Exception:
                        pass
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    else:
                        policy_nn.reset(dones)

                step += 1
                if args_cli.status_every and step % args_cli.status_every == 0:
                    cmd = command_term.vel_command_b[0]
                    vel = robot.data.root_lin_vel_b.torch[0]
                    pos = robot.data.root_pos_w.torch[0]
                    print(
                        f"[{step:6d}] cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f})"
                        f"  vel=({vel[0]:+.2f},{vel[1]:+.2f})"
                        f"  pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f})"
                    )

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)
            env.close()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
