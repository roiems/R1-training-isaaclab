# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_physx.physics import PhysxCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.manager_based.locomotion.velocity import mdp

from .rough_env_cfg import R1RoughEnvCfg, R1Rewards, R1_CFG

@configclass
class PhysicsCfg(PresetCfg):
    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    physx = default

@configclass
class R1FlatRewards(R1Rewards):
    """Reward terms for Unitree R1 Forward Humanoid Locomotion with Real Strides."""
    
    # 1. Primary Objective: Aggressive Forward Velocity Tracking
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": 0.25}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.25}
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.4)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)

    # 2. Upright Torso & Natural Standing Height
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis_link")}
    )
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={"target_height": 0.84, "asset_cfg": SceneEntityCfg("robot", body_names="pelvis_link")},
    )

    # 3. Leg Spacing & Hip Alignment (Prevent Wide Splay)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.35,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # 4. Enforce True Bipedal Walking Steps (Flight Phase)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.2,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.38,
        },
    )

    # 5. Anti-Foot-Slide Penalty
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # 6. Gaze Stabilization & Head Twitch Suppression
    head_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.75,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*head.*"])},
    )
    head_joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-5.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*head.*"])},
    )

    # Upper Body Posture Regularization
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"])},
    )
    joint_deviation_head = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["head_.*_joint"])},
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*"])},
    )

    # 7. Calibrated Smoothness
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.008)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.5e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    
    # 8. Alive & Survival
    is_alive = RewTerm(func=mdp.is_alive, weight=2.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

@configclass
class R1FlatEnvCfg(R1RoughEnvCfg):
    rewards: R1FlatRewards = R1FlatRewards()
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())

    def __post_init__(self):
        super().__post_init__()

        # Flat Plane Terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # Contact Sensor for Feet Air Time & Slide Penalties
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=True,
        )

        # Force Positive Forward Linear Velocity Commands (0.5 - 1.2 m/s)
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)

@configclass
class R1FlatEnvCfg_PLAY(R1FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
