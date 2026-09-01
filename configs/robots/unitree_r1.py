##

R1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/rriemer/Downloads/r1_description/R1/R1.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=20.0,
            max_angular_velocity=20.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.02),
        joint_pos={
            ".*_hip_pitch_joint": -0.26,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.52,
            ".*_ankle_pitch_joint": -0.26,
            ".*_ankle_roll_joint": 0.0,
            "waist_.*_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.20,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.40,
            ".*_wrist_roll_joint": 0.0,
            "head_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint"],
            effort_limit_sim=250.0,
            stiffness={
                ".*_hip_pitch_joint": 180.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_yaw_joint": 150.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_pitch_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_yaw_joint": 5.0,
                ".*_knee_joint": 6.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=100.0,
            stiffness={".*_ankle_.*": 50.0},
            damping={".*_ankle_.*": 4.0},
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit_sim=200.0,
            stiffness={"waist_.*": 150.0},
            damping={"waist_.*": 5.0},
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"],
            effort_limit_sim=100.0,
            stiffness={".*": 40.0},
            damping={".*": 3.0},
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_.*"],
            effort_limit_sim=40.0,
            stiffness={"head_.*": 30.0},
            damping={"head_.*": 12.0},
        ),
    },
)
