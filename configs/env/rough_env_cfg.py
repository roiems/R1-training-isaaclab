# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)
from isaaclab_assets import R1_CFG

# ==============================================================================
# Procedural Obstacle Course Terrain (Boxes, Gaps/Holes, Stairs, Rough Ground)
# ==============================================================================
R1_OBSTACLE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.25,
            grid_width=0.45,
            grid_height_range=(0.04, 0.16),
            platform_width=2.5
        ),
        "gaps_and_holes": terrain_gen.MeshGapTerrainCfg(
            proportion=0.25,
            gap_width_range=(0.15, 0.35),
            platform_width=2.5
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.04, 0.12),
            step_width=0.32,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.02, 0.06),
            noise_step=0.02,
            border_width=0.25
        ),
    },
)

@configclass
class R1Rewards(RewardsCfg):
    """Reward terms for Unitree R1 Humanoid Locomotion MDP."""
    # Positive reward for staying upright & standing each timestep
    is_alive = RewTerm(func=mdp.is_alive, weight=2.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    
    # Base Height Control (prevents knee buckling & crouching)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={"target_height": 0.75}
    )
    
    # Velocity Tracking
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.5, params={"command_name": "base_velocity", "std": 0.5}
    )
    
    # Upright Torso & Roll/Pitch Stability
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    
    # Action Smoothness & Torque Limits
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle_.*")}
    )
    
    # Arm & Waist Natural Balance Posture
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"])},
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint")}
    )

@configclass
class R1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: R1Rewards = R1Rewards()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.vel_yaw_success_threshold = 0.8
        self.scene.robot = R1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner = None
        self.scene.contact_forces = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # Configure Procedural Obstacle Course Terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=R1_OBSTACLE_TERRAIN_CFG,
            max_init_terrain_level=5,
            collision_group=-1,
            debug_vis=False,
        )

        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_com = None
        self.events.base_external_force_torque = None

        self.rewards.undesired_contacts = None
        self.rewards.feet_air_time = None
        self.rewards.feet_slide = None
        self.rewards.dof_torques_l2.weight = 0.0

        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        # Robust biped terminations
        self.terminations.base_contact = None
        self.terminations.root_height_below_minimum = DoneTerm(
            func=mdp.root_height_below_minimum, params={"minimum_height": 0.55}
        )
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation, params={"limit_angle": 0.8}
        )
