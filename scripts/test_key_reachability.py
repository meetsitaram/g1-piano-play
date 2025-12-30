#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test script to visualize piano keys and verify G1 can reach them.

This script spawns G1 robot, table, bench, and piano keys with configurable dimensions.
Use this to verify key positions and reachability before training.

Usage:
    ./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/test_key_reachability.py --num_keys 2
    ./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/test_key_reachability.py --num_keys 7 --key_width 0.1 --key_spacing 0.05
"""

import argparse
import torch

import isaaclab.sim as sim_utils
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test piano key reachability with G1 robot")
parser.add_argument("--num_keys", type=int, default=2, help="Number of piano keys (2, 3, 5, or 7)")
parser.add_argument("--key_width", type=float, default=0.25, help="Width of each key in meters (default: 0.25 for Phase 2.1)")
parser.add_argument("--key_spacing", type=float, default=0.2, help="Spacing between keys in meters (default: 0.2 for Phase 2.1)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch OmniIsaacSim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# Pre-defined robot spawn configurations
# from isaaclab_assets import CRAZYFLIE_CFG, UNITREE_GO2_CFG  # noqa: F401
from isaaclab_assets import UNITREE_G1_CFG

##
# Scene definition
##


@configclass
class KeyReachabilitySceneCfg(InteractiveSceneCfg):
    """Configuration for key reachability test scene."""

    # Ground plane
    ground = sim_utils.GroundPlaneCfg()

    # Lights
    dome_light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    distant_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0), angle=0.5)

    # Robot (G1)
    robot: ArticulationCfg = UNITREE_G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def spawn_table(scene, table_height=0.43):
    """Spawn table using procedural cuboid."""
    table_width = 1.2
    table_depth = 0.6
    table_z = table_height / 2.0

    table_cfg = sim_utils.MeshCuboidCfg(
        size=(table_width, table_depth, table_height),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
    )
    table_cfg.func("/World/envs/env_.*/Table", table_cfg, translation=(0.0, 0.0, table_z))
    print(f"[INFO]: Table spawned at z={table_height:.2f}m (top surface)")
    return table_height


def spawn_bench(scene, bench_height=0.45):
    """Spawn bench for G1 to sit on."""
    bench_width = 0.6
    bench_depth = 0.4
    bench_y = -0.8  # Behind robot
    bench_z = bench_height / 2.0

    bench_cfg = sim_utils.MeshCuboidCfg(
        size=(bench_width, bench_depth, bench_height),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
    )
    bench_cfg.func("/World/envs/env_.*/Bench", bench_cfg, translation=(0.0, bench_y, bench_z))
    print(f"[INFO]: Bench spawned at y={bench_y:.2f}m, z={bench_height:.2f}m")
    return bench_height


def get_key_color(key_index):
    """Return RGB color for each key (visual debugging)."""
    colors = [
        (1.0, 0.0, 0.0),  # C - Red
        (1.0, 0.5, 0.0),  # D - Orange
        (1.0, 1.0, 0.0),  # E - Yellow
        (0.0, 1.0, 0.0),  # F - Green
        (0.0, 0.5, 1.0),  # G - Cyan
        (0.0, 0.0, 1.0),  # A - Blue
        (0.5, 0.0, 1.0),  # B - Purple
    ]
    return colors[key_index]


def spawn_piano_keys(scene, num_keys=2, key_width=0.25, key_spacing=0.2, table_height=0.43):
    """Spawn piano keys with configurable dimensions.
    
    Args:
        scene: The interactive scene
        num_keys: Number of keys to spawn (2, 3, 5, or 7)
        key_width: Width of each key in meters
        key_spacing: Gap between adjacent keys in meters
        table_height: Height of table top
    
    Returns:
        List of key positions (x, y, z) for each key
    """
    key_names_all = ["C", "D", "E", "F", "G", "A", "B"]
    key_names = key_names_all[:num_keys]
    
    key_length = 0.2  # Y dimension
    key_height = 0.1  # Z dimension
    
    # Calculate total width
    total_width = num_keys * key_width + (num_keys - 1) * key_spacing
    
    # Starting position (left edge)
    start_x = -total_width / 2.0
    
    # Key Y and Z positions (same for all keys)
    key_y = -0.21  # Toward robot (table front edge)
    key_z = table_height + key_height / 2.0  # On table top
    
    key_positions = []
    
    print(f"\n[INFO]: Spawning {num_keys} piano keys:")
    print(f"         Key width: {key_width*1000:.0f}mm")
    print(f"         Key spacing: {key_spacing*1000:.0f}mm")
    print(f"         Total piano width: {total_width:.3f}m")
    print(f"         Keys:")
    
    for i, key_name in enumerate(key_names):
        # Calculate key center position
        key_x = start_x + i * (key_width + key_spacing) + key_width / 2.0
        
        # Spawn key
        key_cfg = sim_utils.MeshCuboidCfg(
            size=(key_width, key_length, key_height),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=get_key_color(i)),
        )
        key_cfg.func(
            f"/World/envs/env_.*/PianoKey_{key_name}",
            key_cfg,
            translation=(key_x, key_y, key_z)
        )
        
        key_positions.append((key_x, key_y, key_z))
        print(f"           [{i}] {key_name}: x={key_x:+.3f}m, y={key_y:.2f}m, z={key_z:.2f}m ({get_key_color(i)})")
    
    return key_positions


def set_sitting_pose(robot: Articulation):
    """Set G1 to sitting pose."""
    # Sitting joint positions (from Phase 1)
    sitting_positions = {
        # Legs (sitting)
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
        
        # Arms (slightly forward for testing)
        "left_shoulder_pitch_joint": -0.5,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_pitch_joint": 0.5,  # Note: 0.5 not -0.5 to avoid joint limits
        "left_elbow_roll_joint": 0.0,
        "right_shoulder_pitch_joint": -0.5,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_pitch_joint": 0.5,
        "right_elbow_roll_joint": 0.0,
    }
    
    # Create joint position tensor
    joint_pos = torch.zeros((robot.num_instances, robot.num_joints), device=robot.device)
    for joint_name, pos in sitting_positions.items():
        if joint_name in robot.data.joint_names:
            joint_idx = robot.data.joint_names.index(joint_name)
            joint_pos[:, joint_idx] = pos
    
    # Set joint positions
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    
    print(f"[INFO]: G1 set to sitting pose")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, key_positions: list):
    """Run the simulator and display information."""
    # Set robot pose
    robot_pos = torch.tensor([0.0, -0.8, 0.60], device=sim.device)  # On bench
    robot_rot = torch.tensor([0.707, 0.0, 0.0, 0.707], device=sim.device)  # Face forward
    
    # Set root state
    default_root_state = scene["robot"].data.default_root_state.clone()
    default_root_state[:, :3] += robot_pos
    default_root_state[:, 3:7] = robot_rot
    scene["robot"].write_root_pose_to_sim(default_root_state[:, :7])
    scene["robot"].write_root_velocity_to_sim(default_root_state[:, 7:])
    
    # Set sitting pose
    set_sitting_pose(scene["robot"])
    
    # Simulate physics
    for _ in range(10):
        sim.step()
    
    # Reset buffers
    scene.reset()
    
    print("\n" + "=" * 80)
    print("KEY REACHABILITY TEST")
    print("=" * 80)
    
    print(f"\n[INFO]: Scene setup:")
    print(f"         G1 robot at: (0.0, -0.8, 0.60)")
    print(f"         Table at: (0.0, 0.0, 0.43)")
    print(f"         Bench at: (0.0, -0.8, 0.45)")
    print(f"         Number of keys: {len(key_positions)}")
    
    print(f"\n[INFO]: Key positions (world frame, env 0):")
    key_names = ["C", "D", "E", "F", "G", "A", "B"][:len(key_positions)]
    for i, (x, y, z) in enumerate(key_positions):
        print(f"         {key_names[i]}: ({x:+.3f}, {y:.2f}, {z:.2f})")
    
    # Calculate distances from robot torso to keys
    robot_torso_x = 0.0
    robot_torso_y = -0.8
    
    print(f"\n[INFO]: Distance from G1 torso to keys:")
    for i, (x, y, z) in enumerate(key_positions):
        dist_forward = abs(y - robot_torso_y)
        dist_lateral = abs(x - robot_torso_x)
        dist_total = ((x - robot_torso_x)**2 + (y - robot_torso_y)**2)**0.5
        print(f"         {key_names[i]}: forward={dist_forward:.2f}m, lateral={dist_lateral:.2f}m, total={dist_total:.2f}m")
    
    print(f"\n[INFO]: Estimated G1 reach (sitting):")
    print(f"         Forward: ~0.6-0.8m")
    print(f"         Lateral: ~±0.4m")
    
    # Check reachability
    print(f"\n[INFO]: Reachability assessment:")
    all_reachable = True
    for i, (x, y, z) in enumerate(key_positions):
        dist_forward = abs(y - robot_torso_y)
        dist_lateral = abs(x - robot_torso_x)
        
        forward_ok = dist_forward <= 0.8
        lateral_ok = dist_lateral <= 0.4
        reachable = forward_ok and lateral_ok
        
        status = "✅ REACHABLE" if reachable else "❌ UNREACHABLE"
        print(f"         {key_names[i]}: {status}")
        
        if not reachable:
            all_reachable = False
            if not forward_ok:
                print(f"                  → Too far forward ({dist_forward:.2f}m > 0.8m)")
            if not lateral_ok:
                print(f"                  → Too far lateral ({dist_lateral:.2f}m > 0.4m)")
    
    if all_reachable:
        print(f"\n✅ SUCCESS: All keys appear reachable!")
    else:
        print(f"\n⚠️  WARNING: Some keys may be unreachable. Consider adjusting dimensions.")
    
    print("\n" + "=" * 80)
    print("Simulation running. Close window to exit.")
    print("Observe the robot and keys in the viewport.")
    print("=" * 80 + "\n")
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    # Simulate physics
    while simulation_app.is_running():
        # Reset every 500 steps
        if count % 500 == 0:
            # Reset counters
            sim_time = 0.0
            count = 0
            # Reset robot state
            scene["robot"].write_root_pose_to_sim(default_root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(default_root_state[:, 7:])
            set_sitting_pose(scene["robot"])
            # Reset scene
            scene.reset()
            print("[INFO]: Resetting scene...")
        
        # Apply no actions (robot stays in sitting pose)
        scene["robot"].set_joint_position_target(scene["robot"].data.default_joint_pos)
        
        # Write data to sim
        scene.write_data_to_sim()
        
        # Perform step
        sim.step()
        
        # Increment counters
        sim_time += sim_dt
        count += 1
        
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load simulation configuration
    sim_cfg = SimulationCfg(dt=1/120, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set main camera
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.5])
    
    # Design scene
    scene_cfg = KeyReachabilitySceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    
    # Spawn environment objects
    table_height = spawn_table(scene)
    bench_height = spawn_bench(scene)
    key_positions = spawn_piano_keys(
        scene,
        num_keys=args_cli.num_keys,
        key_width=args_cli.key_width,
        key_spacing=args_cli.key_spacing,
        table_height=table_height
    )
    
    # Play the simulator
    sim.reset()
    
    # Run simulation
    run_simulator(sim, scene, key_positions)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()

