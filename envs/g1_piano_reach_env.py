# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 Piano Reaching RL Environment.

This environment trains the Unitree G1 humanoid robot to reach toward a piano
while sitting on a bench. The robot learns to touch piano targets with both hands.

Phase 1: Simple piano touch task (distance-based contact)
Phase 2 (future): Press specific piano keys (force-based contact)
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
# TODO: Uncomment for Phase 2 (Option B - ContactSensor)
# from isaaclab.sensors import ContactSensor, ContactSensorCfg

from .g1_piano_reach_env_cfg import G1PianoReachEnvCfg


class G1PianoReachEnv(DirectRLEnv):
    """G1 Piano Reaching Environment.
    
    The robot sits on a bench and learns to reach both hands toward piano targets.
    Legs are controlled with fixed sitting pose (not learned).
    Arms are controlled by RL policy.
    """
    
    cfg: G1PianoReachEnvCfg
    
    def __init__(self, cfg: G1PianoReachEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.
        
        Args:
            cfg: Configuration for the environment.
            render_mode: Render mode for the environment.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(cfg, render_mode, **kwargs)
        
        # Get sitting pose (fixed, not trainable)
        self._sitting_joint_positions = self._get_sitting_pose()
        
        # Cache joint indices for fast access
        self._arm_joint_indices, self._arm_joint_names = self._get_arm_joint_indices()
        self._leg_joint_indices, self._leg_joint_names = self._get_leg_joint_indices()
        
        # Cache body indices for hand positions
        self._left_hand_idx, self._right_hand_idx = self._get_hand_body_indices()
        
        # Compute target positions in world frame
        self._compute_target_positions()
        
        # Track previous actions for smoothness reward
        self.previous_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        
        # Track previous arm velocities for acceleration penalty (smoothness)
        self._previous_arm_vel = torch.zeros((self.num_envs, len(self._arm_joint_indices)), device=self.device)
        
        # TODO: Phase 2 (Option B) - Initialize ContactSensor
        # Uncomment when upgrading to contact sensor:
        # if self.cfg.contact_detection_method == "sensor":
        #     self.contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)
        #     self.scene.sensors["hand_contact"] = self.contact_sensor
        
        print(f"[INFO]: G1 Piano Reach Environment initialized")
        print(f"[INFO]:   - Num envs: {self.num_envs}")
        print(f"[INFO]:   - Arm joints: {self._arm_joint_names}")
        print(f"[INFO]:   - Observation dim: {self.cfg.observation_space}")
        print(f"[INFO]:   - Action dim: {self.cfg.action_space}")
        print(f"[INFO]:   - Contact detection: {self.cfg.contact_detection_method}")
    
    # ========================================================================
    # Scene Setup
    # ========================================================================
    
    def _setup_scene(self):
        """Set up the scene with robot, piano, bench, and table.
        
        This reuses logic from the original g1-piano-play.py demo.
        """
        # Spawn G1 robot
        self.robot = Articulation(self.cfg.robot_cfg)
        
        # Spawn ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # Spawn bench (procedural cuboid)
        self._spawn_bench()
        
        # Spawn table
        self._spawn_table()
        
        # Spawn piano
        self._spawn_piano()
        
        # Clone environments for parallelization
        self.scene.clone_environments(copy_from_source=False)
        
        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        
        # Add robot to scene
        self.scene.articulations["robot"] = self.robot
        
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)
    
    def _spawn_bench(self):
        """Spawn bench for robot to sit on (from original demo)."""
        # Bench configuration (from original demo)
        # Height 0.35m, center at 0.275m so top is at 0.45m
        # Position at y=-0.8 (distance from table at y=0)
        bench_distance_from_table = 0.8
        bench_y = -bench_distance_from_table
        bench_z = 0.275  # Center height (top at 0.45m)
        
        bench_cfg = sim_utils.MeshCuboidCfg(
            size=(1.0, 0.35, 0.35),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02,
                rest_offset=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        )
        bench_cfg.func(
            "/World/envs/env_.*/Bench",
            bench_cfg,
            translation=(0.0, bench_y, bench_z)
        )
        print(f"[INFO]: Bench spawned at y={bench_y:.2f}, z={bench_z:.2f} (top at z=0.45)")
    
    def _spawn_table(self):
        """Spawn table for piano (procedural cuboid for reliable cloning)."""
        # Table dimensions
        table_width = 1.2    # X dimension (meters)
        table_depth = 0.6    # Y dimension (meters)
        table_height = 0.43  # Z dimension (meters)
        
        # Position: center of cuboid at z = height/2 to put bottom at z=0
        table_z = table_height / 2.0
        
        table_cfg = sim_utils.MeshCuboidCfg(
            size=(table_width, table_depth, table_height),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,  # Static/immovable
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),  # Brown
        )
        table_cfg.func(
            "/World/envs/env_.*/Table",
            table_cfg,
            translation=(0.0, 0.0, table_z)
        )
        
        # Store table height for piano placement
        self._table_top_height = table_height
        print(f"[INFO]: Procedural table spawned (top at z={table_height:.2f}m)")
    
    def _spawn_piano(self):
        """Spawn piano on table (procedural cuboid for reliable cloning)."""
        # Piano dimensions (keyboard approximation)
        piano_width = 0.5   # X dimension (keyboard width - keeping same size as USD)
        piano_depth = 0.2   # Y dimension (front to back)
        piano_height = 0.1  # Z dimension (keyboard height)
        
        # Position relative to table
        table_top_height = getattr(self, '_table_top_height', 0.43)
        
        # Piano center position
        # Place on table top, toward robot (robot at y=-0.8)
        piano_x = 0.0  # Center on table
        piano_y = -0.21  # Toward robot
        piano_z = table_top_height + piano_height / 2.0  # On table top
        
        piano_cfg = sim_utils.MeshCuboidCfg(
            size=(piano_width, piano_depth, piano_height),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,  # Static/immovable
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),  # Black
        )
        piano_cfg.func(
            "/World/envs/env_.*/Piano",
            piano_cfg,
            translation=(piano_x, piano_y, piano_z)
        )
        
        print(f"[INFO]: Procedural piano spawned at x={piano_x:.2f}, y={piano_y:.2f}, z={piano_z:.3f}m")
        print(f"[INFO]: Table at y=0.0, Piano toward robot at y={piano_y:.2f}m")
    
    # ========================================================================
    # Helper Methods - Joint and Body Indices
    # ========================================================================
    
    def _get_arm_joint_indices(self) -> tuple[torch.Tensor, list[str]]:
        """Get indices of arm joints (shoulder and elbow)."""
        arm_joint_names = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_elbow_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_elbow_roll_joint",
        ]
        
        indices = []
        found_names = []
        for joint_name in arm_joint_names:
            if joint_name in self.robot.data.joint_names:
                idx = self.robot.data.joint_names.index(joint_name)
                indices.append(idx)
                found_names.append(joint_name)
            else:
                print(f"[WARNING]: Joint '{joint_name}' not found in robot")
        
        return torch.tensor(indices, dtype=torch.long, device=self.device), found_names
    
    def _get_leg_joint_indices(self) -> tuple[torch.Tensor, list[str]]:
        """Get indices of leg joints (hip, knee, ankle)."""
        leg_joint_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ]
        
        indices = []
        found_names = []
        for joint_name in leg_joint_names:
            if joint_name in self.robot.data.joint_names:
                idx = self.robot.data.joint_names.index(joint_name)
                indices.append(idx)
                found_names.append(joint_name)
        
        return torch.tensor(indices, dtype=torch.long, device=self.device), found_names
    
    def _get_hand_body_indices(self) -> tuple[int, int]:
        """Get body indices for left and right hands.
        
        Returns:
            Tuple of (left_hand_idx, right_hand_idx)
        """
        # Try common hand/palm body names
        possible_names = ["palm", "hand", "wrist", "gripper"]
        
        left_idx = None
        right_idx = None
        
        for i, body_name in enumerate(self.robot.body_names):
            body_lower = body_name.lower()
            if "left" in body_lower:
                for name in possible_names:
                    if name in body_lower:
                        left_idx = i
                        break
            if "right" in body_lower:
                for name in possible_names:
                    if name in body_lower:
                        right_idx = i
                        break
        
        if left_idx is None or right_idx is None:
            # Fallback: print all body names and raise error
            print("[ERROR]: Could not find hand bodies. Available bodies:")
            for i, name in enumerate(self.robot.body_names):
                print(f"  [{i}] {name}")
            raise ValueError("Could not identify left and right hand bodies")
        
        print(f"[INFO]: Found hand bodies:")
        print(f"         Left hand  (idx {left_idx}): {self.robot.body_names[left_idx]}")
        print(f"         Right hand (idx {right_idx}): {self.robot.body_names[right_idx]}")
        return left_idx, right_idx
    
    def _get_sitting_pose(self) -> torch.Tensor:
        """Get sitting pose joint positions (from original demo).
        
        Returns:
            Tensor of joint positions for sitting pose.
        """
        # Sitting pose (from g1-piano-play.py)
        sitting_positions_dict = {
            # Leg joints
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
            # Arm joints (default comfortable position)
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
        }
        
        # Convert to tensor (only leg joints, arms will be controlled by policy)
        sitting_tensor = torch.zeros(self.robot.num_joints, device=self.device)
        for joint_name, position in sitting_positions_dict.items():
            if joint_name in self.robot.data.joint_names:
                idx = self.robot.data.joint_names.index(joint_name)
                sitting_tensor[idx] = position
        
        return sitting_tensor
    
    def _compute_target_positions(self):
        """Compute target positions for hands in world frame.
        
        NOTE: For parallel environments, we need to compute targets relative to 
        each environment's origin since robots are spatially separated.
        """
        # Get robot root positions for all environments
        # This is called after scene setup, so we can get actual positions
        robot_root_pos = self.robot.data.root_pos_w  # Shape: (num_envs, 3)
        
        # Piano center RELATIVE to robot spawn position
        # Robot spawns at (0, -0.8, 0.6) relative to env origin
        # Piano is at (0, -0.21, 0.48) relative to env origin
        # So piano is at (0, -0.21-(-0.8), 0.48-0.6) = (0, 0.59, -0.12) relative to robot
        
        piano_offset_from_robot = torch.tensor(
            [
                self.cfg.piano_center_x - self.cfg.robot_cfg.init_state.pos[0],
                self.cfg.piano_center_y - self.cfg.robot_cfg.init_state.pos[1],
                self.cfg.piano_center_z - self.cfg.robot_cfg.init_state.pos[2],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        
        # Compute piano center for each environment
        piano_center_per_env = robot_root_pos + piano_offset_from_robot  # Shape: (num_envs, 3)
        
        # Left hand target (offset from piano center)
        left_offset = torch.tensor(
            self.cfg.left_hand_target_offset,
            device=self.device,
            dtype=torch.float32,
        )
        self.left_hand_target_w = piano_center_per_env + left_offset
        
        # Right hand target (offset from piano center)
        right_offset = torch.tensor(
            self.cfg.right_hand_target_offset,
            device=self.device,
            dtype=torch.float32,
        )
        self.right_hand_target_w = piano_center_per_env + right_offset
        
        print(f"[INFO]: Left hand target (env 0): {self.left_hand_target_w[0]}")
        print(f"[INFO]: Right hand target (env 0): {self.right_hand_target_w[0]}")
        print(f"[INFO]: Piano offset from robot: {piano_offset_from_robot}")
        print(f"[INFO]: Robot root position (env 0): {robot_root_pos[0]}")
        print(f"[INFO]: Robot spawn position (config): ({self.cfg.robot_cfg.init_state.pos})")
        print(f"[INFO]: Robot initial position: {self.cfg.robot_cfg.init_state.pos}")
    
    # ========================================================================
    # MDP Methods - Core RL Loop
    # ========================================================================
    
    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step.
        
        Args:
            actions: Raw actions from policy (num_envs, action_dim)
        """
        # Scale actions
        self.actions = self.cfg.action_scale * actions.clone()
    
    def _apply_action(self):
        """Apply actions to robot.
        
        Arms are controlled by RL policy.
        Legs are controlled by fixed sitting pose.
        """
        # Apply RL actions to arm joints
        self.robot.set_joint_position_target(
            self.actions,
            joint_ids=self._arm_joint_indices,
        )
        
        # Apply fixed sitting pose to leg joints
        leg_targets = self._sitting_joint_positions[self._leg_joint_indices].unsqueeze(0).expand(self.num_envs, -1)
        self.robot.set_joint_position_target(
            leg_targets,
            joint_ids=self._leg_joint_indices,
        )
    
    def _get_observations(self) -> dict:
        """Compute observations for policy.
        
        Observation space (36 dims - MINIMAL as discussed):
        - Arm joint positions (10)
        - Arm joint velocities (10)
        - Left hand to target (3)
        - Right hand to target (3)
        - Previous actions (10)
        
        Returns:
            Dictionary with "policy" key containing observations.
        """
        # Get arm joint states
        arm_pos = self.robot.data.joint_pos[:, self._arm_joint_indices]  # (num_envs, 10)
        arm_vel = self.robot.data.joint_vel[:, self._arm_joint_indices]  # (num_envs, 10)
        
        # Get hand positions in world frame
        left_hand_pos = self.robot.data.body_pos_w[:, self._left_hand_idx]  # (num_envs, 3)
        right_hand_pos = self.robot.data.body_pos_w[:, self._right_hand_idx]  # (num_envs, 3)
        
        # Compute hand-to-target vectors
        left_hand_to_target = left_hand_pos - self.left_hand_target_w  # (num_envs, 3)
        right_hand_to_target = right_hand_pos - self.right_hand_target_w  # (num_envs, 3)
        
        # Concatenate all observations
        obs = torch.cat(
            [
                arm_pos,                    # 10
                arm_vel,                    # 10
                left_hand_to_target,        # 3
                right_hand_to_target,       # 3
                self.previous_actions,      # 10
            ],
            dim=-1,
        )  # Total: 36
        
        # Update previous actions
        self.previous_actions = self.actions.clone()
        
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards.
        
        Reward components:
        1. Distance reward: Encourage moving hands toward targets
        2. Contact reward: Bonus for touching piano (distance or force-based)
        3. Both hands bonus: Extra reward when both hands touch
        4. Action smoothness: Penalize jerky motions
        5. Joint limit penalty: Penalize approaching joint limits
        
        Returns:
            Reward tensor (num_envs,)
        """
        # Get hand positions
        left_hand_pos = self.robot.data.body_pos_w[:, self._left_hand_idx]
        right_hand_pos = self.robot.data.body_pos_w[:, self._right_hand_idx]
        
        # === Distance Reward ===
        left_dist = torch.norm(left_hand_pos - self.left_hand_target_w, dim=-1)
        right_dist = torch.norm(right_hand_pos - self.right_hand_target_w, dim=-1)
        dist_reward = -self.cfg.rew_scale_reaching * (left_dist + right_dist)
        
        # Debug logging removed for training performance
        
        # === Contact Detection ===
        if self.cfg.contact_detection_method == "distance":
            # Option A: Distance-based contact (DEFAULT)
            left_contact = (left_dist < self.cfg.contact_distance_threshold).float()
            right_contact = (right_dist < self.cfg.contact_distance_threshold).float()
        else:
            # TODO: Option B: Force-based contact (Phase 2)
            # Uncomment when upgrading to ContactSensor:
            # contact_forces = self._get_hand_contact_forces()
            # left_contact = (contact_forces[:, 0] > self.cfg.contact_force_threshold).float()
            # right_contact = (contact_forces[:, 1] > self.cfg.contact_force_threshold).float()
            
            # Fallback to distance for now
            left_contact = (left_dist < self.cfg.contact_distance_threshold).float()
            right_contact = (right_dist < self.cfg.contact_distance_threshold).float()
        
        # Contact rewards
        contact_reward = self.cfg.rew_scale_contact * (left_contact + right_contact)
        both_hands_reward = self.cfg.rew_scale_both_hands * (left_contact * right_contact)
        
        # === Action Smoothness ===
        # Penalize large changes in actions between timesteps (reduces flapping)
        action_diff = torch.sum(torch.square(self.actions - self.previous_actions), dim=-1)
        action_penalty = self.cfg.rew_scale_action_rate * action_diff
        
        # === Joint Velocity Penalty ===
        # Penalize high arm joint velocities for smoother motion
        arm_vel = self.robot.data.joint_vel[:, self._arm_joint_indices]
        joint_vel_penalty = self.cfg.rew_scale_joint_vel * torch.sum(torch.square(arm_vel), dim=-1)
        
        # === Joint Acceleration Penalty (Optional) ===
        # Penalize rapid velocity changes for even smoother motion
        if hasattr(self, '_previous_arm_vel'):
            arm_accel = arm_vel - self._previous_arm_vel
            joint_accel_penalty = self.cfg.rew_scale_joint_accel * torch.sum(torch.square(arm_accel), dim=-1)
        else:
            joint_accel_penalty = torch.zeros(self.num_envs, device=self.device)
        
        # Store velocity for next step (for acceleration calculation)
        self._previous_arm_vel = arm_vel.clone()
        
        # === Joint Limit Penalty ===
        # Get normalized joint positions (assuming symmetric limits around 0)
        arm_pos = self.robot.data.joint_pos[:, self._arm_joint_indices]
        # Simple limit check: penalize when |pos| > 0.9 of assumed max range (e.g., 2.0 rad)
        near_limits = torch.sum(torch.square(torch.clamp(torch.abs(arm_pos) - 1.8, min=0.0)), dim=-1)
        joint_limit_penalty = self.cfg.rew_scale_joint_limit * near_limits
        
        # === Total Reward ===
        total_reward = (
            dist_reward +
            contact_reward +
            both_hands_reward +
            action_penalty +
            joint_vel_penalty +
            joint_accel_penalty +
            joint_limit_penalty
        )
        
        return total_reward
    
    # TODO: Phase 2 (Option B) - Contact force extraction
    # Uncomment when upgrading to ContactSensor:
    # def _get_hand_contact_forces(self) -> torch.Tensor:
    #     """Get contact forces on hands from ContactSensor.
    #     
    #     Returns:
    #         Tensor of shape (num_envs, 2) with [left_force, right_force]
    #     """
    #     contact_data = self.contact_sensor.data
    #     left_force = contact_data.net_forces_w[:, 0].norm(dim=-1)
    #     right_force = contact_data.net_forces_w[:, 1].norm(dim=-1)
    #     return torch.stack([left_force, right_force], dim=-1)
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions.
        
        Termination conditions:
        1. Torso too low (robot fell)
        2. Hands too high (robot standing up)
        3. Time-out (episode length exceeded)
        
        Returns:
            Tuple of (terminated, truncated) boolean tensors
        """
        # Time-out condition
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Robot fell (torso too low)
        torso_too_low = self.robot.data.root_pos_w[:, 2] < self.cfg.min_torso_height
        
        # Robot standing up (hands too high)
        left_hand_pos = self.robot.data.body_pos_w[:, self._left_hand_idx]
        right_hand_pos = self.robot.data.body_pos_w[:, self._right_hand_idx]
        hands_too_high = torch.any(
            torch.stack([left_hand_pos[:, 2], right_hand_pos[:, 2]], dim=1) > self.cfg.max_hand_height,
            dim=1
        )
        
        # Combine termination conditions
        terminated = torso_too_low | hands_too_high
        
        return terminated, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset environments.
        
        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        super()._reset_idx(env_ids)
        
        # Reset robot to sitting pose with small randomization
        joint_pos = self._sitting_joint_positions.unsqueeze(0).expand(len(env_ids), -1).clone()
        
        # Add small random noise to arm joints only (not legs)
        arm_noise = torch.randn((len(env_ids), len(self._arm_joint_indices)), device=self.device) * 0.1
        joint_pos[:, self._arm_joint_indices] += arm_noise
        
        # Clamp to safe ranges (simple clamp, can use actual limits later)
        joint_pos = torch.clamp(joint_pos, -2.5, 2.5)
        
        joint_vel = torch.zeros_like(joint_pos)
        
        # Reset root state
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        # Write states to simulation
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Reset tracking variables
        self.previous_actions[env_ids] = 0.0
        self._previous_arm_vel[env_ids] = 0.0  # Reset arm velocities for smoothness tracking

