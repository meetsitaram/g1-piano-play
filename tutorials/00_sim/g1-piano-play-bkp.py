# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/spawn_prims.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch
import isaaclab.sim as sim_utils
import isaaclab.sim.utils.prims as prim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# pxr is available through Isaac Sim when using the isaaclab.sh launcher
from pxr import Usd, UsdPhysics, PhysxSchema


def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files.
    
    Height configuration (all Z values):
    - Ground: 0m
    - Table bottom: 0m (configured via table_bottom_in_model parameter)
    - Table top: Calculated as table_bottom_in_model + table_height
    - Piano: Sits on table top (auto-calculated from table height)
    - Bench top: 0.45m
    - Robot base: 0.85m (will sit on bench at 0.45m)
    
    TUNABLE PARAMETERS:
    - table_bottom_in_model: Distance from USD origin to table bottom
    - table_height: Total height of the table
    - piano_y_offset: Move piano along Y axis (negative = towards robot)
    - bench_distance_from_table: Distance of bench from table
    - robot_offset_from_bench: Robot's offset from bench (negative = closer to table)
    
    Robot pose is set in two stages:
    1. get_sitting_joint_positions() - legs bent, body leaning forward
    2. get_arm_reaching_positions() - arms stretched forward to reach piano keys
    """
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # create a new xform prim for all objects to be spawned under
    prim_utils.create_prim("/World/Objects", "Xform")

    # ========================================================================
    # SPAWN TABLE - ADJUST THESE VALUES IF TABLE IS FLOATING OR BURIED
    # ========================================================================
    # If table is buried: INCREASE table_bottom_in_model
    # If table is floating: DECREASE table_bottom_in_model
    # If piano is too high/low: ADJUST table_height
    import os
    
    # TUNABLE PARAMETERS
    table_bottom_in_model = 0.4  # Distance from model origin to table bottom (meters)
    table_height = 0.43           # Total table height (meters)
    
    # Calculate translation needed to put table bottom at ground level (z=0)
    table_z_offset = table_bottom_in_model  # Lift table so bottom touches ground
    # table_top_height = table_z_offset + table_height  # Where the table top will be
    table_top_height = table_height  # Where the table top will be # todo - check this
    
    print(f"[INFO]: Table config - bottom_offset: {table_bottom_in_model}m, height: {table_height}m")
    print(f"[INFO]: Table will be placed with bottom at z=0, top at z={table_top_height:.2f}m")
    
    # Try physics configuration first, fallback to base USD
    table_physics_path = "/home/solotech007/RoboGym/simulation/g1-piano-play/onshape-assets/table/table/configuration/table_physics.usd"
    table_base_path = "/home/solotech007/RoboGym/simulation/g1-piano-play/onshape-assets/table/table/table.usd"
    
    if os.path.exists(table_physics_path):
        table_path = table_physics_path
        print(f"[INFO]: Using table_physics.usd")
    elif os.path.exists(table_base_path):
        table_path = table_base_path
        print(f"[INFO]: Using base table.usd (physics config not found)")
    else:
        print(f"[ERROR]: No table USD file found!")
        table_path = table_base_path  # Try anyway
    
    cfg = sim_utils.UsdFileCfg(
        usd_path=table_path,
        # Add rigid body props to make it static, but don't modify collision mesh
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,  # Make it static/immovable
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),  # wooden brown color
    )
    try:
        cfg.func("/World/Objects/Table", cfg, translation=(0.0, 0.0, table_z_offset))
        print(f"[INFO]: Table spawned at z={table_z_offset}m")
    except Exception as e:
        print(f"[ERROR]: Failed to spawn table: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # SPAWN PIANO - Position on table
    # ========================================================================
    # piano's geometric center is at (0.483, 0.03, -0.0177) from its origin
    # to center it on table (at 0,0), offset by (-0.483, -0.03)
    # Robot is at y=-0.7, so move piano in -Y direction to bring it closer
    
    # TUNABLE PARAMETERS
    piano_y_offset = -0.18  # Move piano towards robot (negative = closer to robot at y=-0.7)
    
    # Calculate piano position
    piano_bottom_offset = 0.0755  # Piano bottom is 0.0755m below its origin
    piano_z = table_top_height + piano_bottom_offset
    piano_x = -0.483  # Center on table in X
    piano_y = -0.030 + piano_y_offset  # Adjust Y position
    
    cfg = sim_utils.UsdFileCfg(
        usd_path=f"/home/solotech007/RoboGym/simulation/g1-piano-play/onshape-assets/piano/piano/piano.usd",
        # Don't override collision props - the USD has instanced collision meshes
    )
    cfg.func("/World/Objects/Piano", cfg, translation=(piano_x, piano_y, piano_z))
    print(f"[INFO]: Piano placed at (x={piano_x:.3f}, y={piano_y:.3f}, z={piano_z:.4f}m)")
    print(f"[INFO]: Piano is on table top at z={table_top_height:.2f}m, moved {piano_y_offset:.2f}m towards robot")

    # ========================================================================
    # SPAWN BENCH - Position for robot to sit on
    # ========================================================================
    # Bench height 0.35m, center at 0.275m so top is at 0.45m
    # Move bench closer to table by adjusting Y position
    
    # TUNABLE PARAMETER
    bench_distance_from_table = 0.8  # Distance from table (positive Y is away from table at y=0)
    
    bench_y = -bench_distance_from_table  # Negative because bench is behind table
    bench_z = 0.275  # Center height to put top at 0.45m
    
    cfg = sim_utils.MeshCuboidCfg(
        size=(1.0, 0.35, 0.35),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,  # Increase contact distance for better collision detection
            rest_offset=0.0,      # Minimum separation distance to prevent sinking
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,  # Allow faster penetration resolution
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=50.0),  # Heavy bench won't move
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
    )
    cfg.func("/World/Objects/Bench", cfg, translation=(0.0, bench_y, bench_z))
    print(f"[INFO]: Bench placed at y={bench_y:.2f}m (distance from table: {bench_distance_from_table:.2f}m)")

    # ========================================================================
    # SPAWN G1 ROBOT - Position on bench
    # ========================================================================
    # Robot positioned relative to bench, will sit down onto it
    # orientation: rotate 90 degrees around Z axis to face the table (positive Y direction)
    
    # TUNABLE PARAMETER
    robot_offset_from_bench = 0.2  # Offset from bench position (negative = closer to table)
    
    robot_y = bench_y + robot_offset_from_bench  # Place robot relative to bench
    robot_z = 0.85     # Position robot above bench (will sit down to 0.45m bench top)
    
    cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_minimal.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,  # Increased to resolve penetration faster
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,  # Increased for better collision resolution
            solver_velocity_iteration_count=8,   # Increased for stability
        ),
    )
    cfg.func("/World/Objects/G1_Robot", cfg, translation=(0.0, robot_y, robot_z), orientation=(0.707, 0.0, 0.0, 0.707))
    print(f"[INFO]: Robot placed at y={robot_y:.2f}m, z={robot_z:.2f}m")
    if robot_offset_from_bench != 0:
        print(f"[INFO]: Robot offset from bench: {robot_offset_from_bench:+.2f}m ({'closer to table' if robot_offset_from_bench > 0 else 'away from table'})")
    print(f"[INFO]: Distance from robot to table edge: {abs(robot_y):.2f}m")


def get_sitting_joint_positions():
    """
    Reference function: Returns pre-calculated sitting pose joint positions.
    
    These values were manually tuned for the G1 robot to sit comfortably on a bench
    at 0.45m height while facing a table/piano at approximately 0.5m height (updated table).
    
    NOTE: The G1 minimal model only has torso_joint which controls YAW (rotation), not PITCH.
    Forward lean is achieved by increasing the hip_pitch angles beyond 90 degrees.
    
    TUNABLE PARAMETERS:
    
    Forward lean (hip_pitch_joint):
    - More negative (e.g., -2.1) = more forward lean
    - Less negative (e.g., -1.8) = more upright
    - -1.57 (~-90°) = sitting straight up
    
    Foot position (ankle_pitch_joint):
    - More positive (e.g., 0.4, 0.5) = tilt foot forward (ball touches ground)
    - Less positive (e.g., 0.1, 0.2) = less tilt
    - Negative values = tilt backward (heel only on ground)
    
    Returns:
        dict: Joint names mapped to positions in radians
    """
    sitting_joint_positions = {
        # Leg joints - primary sitting configuration
        # Since G1 minimal has no torso pitch joint, we increase hip pitch to lean the whole body forward
        "left_hip_pitch_joint": -2,        # ~-115 degrees - forward at hip to lean body forward
        "left_knee_joint": 1.7,            # ~97 degrees - bend knee forward
        "left_ankle_pitch_joint": 0.3,     # ~17 degrees - tilt foot forward so ball touches ground (TUNABLE)
        "right_hip_pitch_joint": -2,       # ~-115 degrees - forward at hip to lean body forward
        "right_knee_joint": 1.7,           # ~97 degrees - bend knee forward
        "right_ankle_pitch_joint": 0.3,    # ~17 degrees - tilt foot forward so ball touches ground (TUNABLE)
        
        # Hip roll - slight leg spread for stability
        "left_hip_roll_joint": 0.1,        # ~6 degrees outward
        "right_hip_roll_joint": -0.1,      # ~6 degrees outward
        
        # Ankle roll - keep feet flat (no tilting left/right)
        "left_ankle_roll_joint": 0.0,      # No side tilt
        "right_ankle_roll_joint": 0.0,     # No side tilt
        
        # Hip yaw - keep legs straight (no rotation)
        "left_hip_yaw_joint": 0.0,         # No rotation
        "right_hip_yaw_joint": 0.0,        # No rotation
        
        # Torso - G1 minimal only has torso_joint which controls YAW (rotation), not PITCH (lean)
        # Since there's no torso pitch joint, we achieve forward lean by adjusting hip angles
        # Set torso_joint to 0 to keep upper body facing forward (no rotation)
        "torso_joint": 0.0,                # Keep torso facing forward (no rotation)
        
        # Arm joints - positioned for piano playing
        "left_shoulder_pitch_joint": 0.5,   # ~29 degrees forward
        "left_shoulder_roll_joint": 0.3,    # ~17 degrees outward
        "left_shoulder_yaw_joint": 0.0,     # No rotation
        "left_elbow_pitch_joint": -0.5,     # ~-29 degrees bend (CORRECTED NAME)
        "left_elbow_roll_joint": 0.0,       # No rotation
        "right_shoulder_pitch_joint": 0.5,  # ~29 degrees forward
        "right_shoulder_roll_joint": -0.3,  # ~17 degrees outward
        "right_shoulder_yaw_joint": 0.0,    # No rotation
        "right_elbow_pitch_joint": -0.5,    # ~-29 degrees bend (CORRECTED NAME)
        "right_elbow_roll_joint": 0.0,      # No rotation
    }
    return sitting_joint_positions


def get_arm_reaching_positions():
    """
    Returns ONLY arm joint positions for reaching forward to piano keys.
    
    This should be applied AFTER the sitting pose has settled.
    Only includes arm joints, so it won't affect the sitting posture.
    
    TUNABLE PARAMETERS:
    - shoulder_pitch: More negative = arms reach higher/more forward
    - elbow_pitch: Closer to 0 = straighter arms (fully extended)
    - shoulder_roll: Controls arm width (how far apart)
    
    Returns:
        dict: Joint names mapped to positions in radians (arms only)
    """
    arm_positions = {
        # Left arm - stretched forward towards piano
        "left_shoulder_pitch_joint": -0.5,   # ~-29 degrees - reach forward (TUNABLE: -0.3 to -0.8)
        "left_shoulder_roll_joint": 0.15,    # ~9 degrees - slight outward (arms not too wide)
        "left_shoulder_yaw_joint": 0.0,      # No rotation
        "left_elbow_pitch_joint": -0.2,      # ~-11 degrees - nearly straight (TUNABLE: 0 to -0.5)
        "left_elbow_roll_joint": 0.0,        # No rotation
        
        # Right arm - stretched forward towards piano
        "right_shoulder_pitch_joint": -0.5,  # ~-29 degrees - reach forward (TUNABLE: -0.3 to -0.8)
        "right_shoulder_roll_joint": -0.15,  # ~-9 degrees - slight outward (arms not too wide)
        "right_shoulder_yaw_joint": 0.0,     # No rotation
        "right_elbow_pitch_joint": -0.2,     # ~-11 degrees - nearly straight (TUNABLE: 0 to -0.5)
        "right_elbow_roll_joint": 0.0,       # No rotation
    }
    
    return arm_positions


def apply_joint_positions(robot, joint_targets, positions_dict, dof_names, sim, sim_dt, 
                          settle_steps=200, description="joint positions"):
    """
    Apply joint positions to the robot and let them settle.
    
    Args:
        robot: Articulation object
        joint_targets: Tensor of joint targets to modify
        positions_dict: Dictionary of joint_name -> position (radians)
        dof_names: List of all joint names on the robot
        sim: SimulationContext
        sim_dt: Simulation timestep
        settle_steps: Number of steps to settle (default: 200)
        description: Description of what's being applied (for logging)
    
    Returns:
        int: Number of joints successfully set
    """
    print(f"[INFO]: Applying {description}...")
    
    num_set = 0
    for joint_name, position in positions_dict.items():
        if joint_name in dof_names:
            idx = dof_names.index(joint_name)
            joint_targets[0, idx] = position
            num_set += 1
            print(f"  Set {joint_name} to {position:.3f} rad")
        else:
            print(f"  [WARNING] Joint '{joint_name}' not found in robot")
    
    print(f"[INFO]: Applied {num_set}/{len(positions_dict)} {description}")
    
    # Let the pose settle
    print(f"[INFO]: Settling {description}...")
    for i in range(settle_steps):
        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        
        # Print progress every 50 steps
        if (i + 1) % 50 == 0:
            print(f"  Step {i+1}/{settle_steps}...")
    
    print(f"[INFO]: ✓ {description.capitalize()} settled!")
    return num_set


def apply_sitting_pose(robot, joint_targets, dof_names, sim, sim_dt):
    """
    Apply sitting pose to the robot.
    
    Args:
        robot: Articulation object
        joint_targets: Tensor of joint targets to modify
        dof_names: List of all joint names on the robot
        sim: SimulationContext
        sim_dt: Simulation timestep
    """
    sitting_positions = get_sitting_joint_positions()
    apply_joint_positions(
        robot, joint_targets, sitting_positions, dof_names, sim, sim_dt,
        settle_steps=200, description="sitting pose"
    )


def apply_arm_reaching_pose(robot, joint_targets, dof_names, sim, sim_dt):
    """
    Apply arm reaching pose to the robot (arms stretched forward to piano).
    
    Args:
        robot: Articulation object
        joint_targets: Tensor of joint targets to modify
        dof_names: List of all joint names on the robot
        sim: SimulationContext
        sim_dt: Simulation timestep
    """
    arm_positions = get_arm_reaching_positions()
    apply_joint_positions(
        robot, joint_targets, arm_positions, dof_names, sim, sim_dt,
        settle_steps=200, description="arm reaching pose"
    )


def check_articulation_root():
    """Check if the G1 robot has ArticulationRootAPI properly applied."""
    stage = prim_utils.get_current_stage()
    robot_prim_path = "/World/Objects/G1_Robot"
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    
    if not robot_prim.IsValid():
        print(f"[WARNING]: Robot prim not found at {robot_prim_path}")
        return False
    
    # Check if this prim has ArticulationRootAPI
    if robot_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        print(f"[INFO]: ArticulationRootAPI found on {robot_prim_path}")
        
        # Check PhysxArticulationAPI
        if robot_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            print(f"[INFO]: PhysxArticulationAPI also applied")
        else:
            print(f"[WARNING]: PhysxArticulationAPI NOT applied")
        return True
    else:
        # Search for ArticulationRootAPI in children
        print(f"[INFO]: Searching for ArticulationRootAPI in children...")
        for prim in Usd.PrimRange(robot_prim):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                print(f"[INFO]: Found ArticulationRootAPI at: {prim.GetPath()}")
                return True
        
        print(f"[WARNING]: No ArticulationRootAPI found in robot hierarchy")
        return False


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
    # Design scene
    design_scene()
    
    # Create Articulation instance for the robot BEFORE sim.reset()
    print("[INFO]: Creating Articulation for the G1 robot...")
    robot_cfg = ArticulationCfg(
        prim_path="/World/Objects/G1_Robot",
        spawn=None,  # Robot already spawned
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"],
                stiffness=400.0,
                damping=40.0,
            ),
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],  # All remaining joints
                stiffness=200.0,
                damping=20.0,
            ),
        },
    )
    robot = Articulation(robot_cfg)
    
    # Play the simulator - this initializes physics
    sim.reset()
    
    # Check if table was spawned
    print("[INFO]: Checking if table was spawned...")
    stage = prim_utils.get_current_stage()
    table_prim = stage.GetPrimAtPath("/World/Objects/Table")
    if table_prim.IsValid():
        print(f"[INFO]: Table prim found at /World/Objects/Table")
        print(f"[INFO]: Table prim type: {table_prim.GetTypeName()}")
        # Check for children
        children = list(table_prim.GetChildren())
        print(f"[INFO]: Table has {len(children)} children: {[c.GetName() for c in children[:5]]}")
    else:
        print("[ERROR]: Table prim NOT found at /World/Objects/Table!")
    
    # Check articulation root configuration
    print("[INFO]: Checking articulation root configuration...")
    check_articulation_root()
    
    print(f"[INFO]: Robot initialized with {robot.num_joints} DOFs")
    print(f"[INFO]: Joint names: {robot.data.joint_names}")
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    
    # Let physics settle briefly
    print("[INFO]: Letting physics settle...")
    for _ in range(10):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
    
    # Get current joint position targets
    joint_targets = robot.data.joint_pos.clone()
    
    # Map joint names to indices
    dof_names = robot.data.joint_names
    
    # Print ALL available joint names for reference
    print(f"\n[DEBUG]: Available joint names on G1 robot:")
    for i, name in enumerate(dof_names):
        print(f"  [{i:2d}] {name}")
    print("-" * 60)
    
    # ===== STEP 1: Apply sitting pose =====
    print("\n[INFO]: STEP 1 - Sitting Pose")
    print("=" * 60)
    apply_sitting_pose(robot, joint_targets, dof_names, sim, sim_dt)
    
    # ===== STEP 2: Stretch arms forward to reach piano =====
    print("\n[INFO]: STEP 2 - Arm Reaching Pose")
    print("=" * 60)
    apply_arm_reaching_pose(robot, joint_targets, dof_names, sim, sim_dt)
    
    # Setup complete!
    print("\n" + "=" * 60)
    print("[INFO]: ✓ Setup complete! Robot is sitting with arms reaching forward.")
    print("=" * 60)
    
    # Simulation loop - maintain piano playing pose (sitting + arms reaching)
    while simulation_app.is_running():
        # Apply joint targets to maintain piano playing pose
        robot.set_joint_position_target(joint_targets)
        # Write data to simulation
        robot.write_data_to_sim()
        # Step simulation
        sim.step()
        # Update robot state
        robot.update(sim_dt)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
