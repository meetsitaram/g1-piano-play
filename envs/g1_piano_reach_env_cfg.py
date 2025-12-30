# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for G1 Piano Reaching RL environment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


@configclass
class G1PianoReachEnvCfg(DirectRLEnvCfg):
    """Configuration for G1 Piano Reaching environment.
    
    The robot sits on a bench and learns to reach toward a piano with both hands.
    
    Phase 1: Touch piano with both hands
    Phase 2 (future): Press specific keys
    """
    
    # ========================================================================
    # Environment Settings
    # ========================================================================
    
    # Simulation stepping
    decimation = 2  # Control at 60Hz (120Hz physics / 2)
    episode_length_s = 10.0  # 10 second episodes
    
    # Action space: 10 DOF (5 per arm)
    # Arms: left/right × [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_roll]
    action_space = 10
    action_scale = 1.0  # No scaling - direct joint position commands
    
    # Observation space: 36 dimensions (MINIMAL - as discussed)
    # - Arm joint positions (10)
    # - Arm joint velocities (10)
    # - Left hand to target (3)
    # - Right hand to target (3)
    # - Previous action (10)
    observation_space = 36
    
    state_space = 0  # No asymmetric observations
    
    # ========================================================================
    # Simulation
    # ========================================================================
    
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,  # 120Hz physics
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_found_lost_pairs_capacity=2**24,
        ),
    )
    
    # ========================================================================
    # Scene
    # ========================================================================
    
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,  # Increased to 2048 parallel environments for faster training
        env_spacing=3.0,  # 3 meters between environments
        replicate_physics=True,
        clone_in_fabric=True,  # Fast cloning
    )
    
    # ========================================================================
    # Robot Configuration
    # ========================================================================
    
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/G1_Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_minimal.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -0.8, 0.60),  # Position on bench (y=-0.8 same as bench, z=0.60 just above bench top at 0.45)
            rot=(0.707, 0.0, 0.0, 0.707),  # 90° rotation to face piano (forward = +Y)
            # Initialize with sitting joint positions for stability
            joint_pos={
                # Legs - sitting position
                "left_hip_pitch_joint": -2.0,
                "left_knee_joint": 1.7,
                "left_ankle_pitch_joint": 0.3,
                "right_hip_pitch_joint": -2.0,
                "right_knee_joint": 1.7,
                "right_ankle_pitch_joint": 0.3,
                "left_hip_roll_joint": 0.1,
                "right_hip_roll_joint": -0.1,
                "left_ankle_roll_joint": 0.0,
                "right_ankle_roll_joint": 0.0,
                "left_hip_yaw_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "torso_joint": 0.0,
                # Arms - neutral starting position
                "left_shoulder_pitch_joint": 0.5,
                "left_shoulder_roll_joint": 0.3,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": 0.5,  # Within limits [-0.227, 3.421]
                "left_elbow_roll_joint": 0.0,
                "right_shoulder_pitch_joint": 0.5,
                "right_shoulder_roll_joint": -0.3,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": 0.5,  # Within limits [-0.227, 3.421]
                "right_elbow_roll_joint": 0.0,
            },
        ),
        actuators={
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*shoulder.*", ".*elbow.*"],
                stiffness=2000.0,  # Increased 10x for stronger position control
                damping=200.0,     # Increased proportionally
                effort_limit=300.0,
                velocity_limit=100.0,
            ),
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"],
                stiffness=400.0,
                damping=40.0,
                effort_limit=300.0,
                velocity_limit=100.0,
            ),
        },
    )
    
    # ========================================================================
    # Piano Target Configuration
    # ========================================================================
    
    # Piano position (procedural cuboid)
    # Table top at z=0.43, piano height=0.1, so center at z=0.43+0.05=0.48
    piano_center_x = 0.0     # Centered on table
    piano_center_y = -0.21   # Toward robot (robot at y=-0.8)
    piano_center_z = 0.48    # Table top (0.43) + half piano height (0.05)
    
    # Target points on piano keyboard (where hands should touch)
    # Offsets relative to piano center (piano width=0.5m, so ±0.2 is within bounds)
    left_hand_target_offset = [-0.20, 0.0, 0.0]   # Left side of keyboard
    right_hand_target_offset = [0.20, 0.0, 0.0]   # Right side of keyboard
    
    # ========================================================================
    # Reward Scales
    # ========================================================================
    
    # Distance reward (encourage moving toward piano)
    rew_scale_reaching = 5.0  # INCREASED from 2.0 - must dominate smoothness penalties!
    
    # Contact reward (bonus for touching piano)
    rew_scale_contact = 15.0  # INCREASED from 10.0 - strong motivation to reach
    
    # Both hands bonus (extra reward when both hands touch)
    rew_scale_both_hands = 10.0  # INCREASED from 5.0 - reward coordination
    
    # === SMOOTHNESS PENALTIES (reduce flapping/flickering) ===
    # Action smoothness (penalize jerky motion)
    rew_scale_action_rate = -0.5  # REDUCED from -1.0 - still discourages flapping but allows reaching
    
    # Joint velocity penalty (penalize high arm velocities for smoother motion)
    rew_scale_joint_vel = -0.05  # Penalize rapid joint movements
    
    # Joint acceleration penalty (penalize rapid velocity changes)
    rew_scale_joint_accel = -0.01  # Penalize acceleration spikes (optional, can disable if needed)
    
    # Joint limits (penalize approaching limits)
    rew_scale_joint_limit = -0.5
    
    # ========================================================================
    # Contact Detection Configuration
    # ========================================================================
    
    # Contact detection method
    # - "distance": Simple distance-based (faster, good for Phase 1) [DEFAULT]
    # - "sensor": Use ContactSensor API (accurate, needed for Phase 2) [TODO]
    contact_detection_method = "distance"
    
    # Distance-based contact threshold (Option A - DEFAULT)
    contact_distance_threshold = 0.05  # 5cm - close enough = "touching"
    
    # TODO: ContactSensor configuration (Option B - for Phase 2)
    # Uncomment and configure when upgrading to ContactSensor:
    # contact_sensor_cfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/G1_Robot/.*palm.*",
    #     update_period=0.0,
    #     history_length=1,
    #     track_air_time=False,
    # )
    # contact_force_threshold = 5.0  # Newtons - actual force threshold
    
    # ========================================================================
    # Termination Conditions
    # ========================================================================
    
    # Termination conditions
    max_hand_height = 1.5  # Reset if hands go too high (standing up)
    min_torso_height = 0.3  # Reset if robot falls (torso too low)
    
    # Success bonus
    success_bonus = 20.0  # Bonus when both hands touch piano

