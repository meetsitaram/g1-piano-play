#!/usr/bin/env python3
"""
Helper script to extract piano dimensions from USD file.
Run with: ./isaaclab.sh -p g1-piano-play/tutorials/00_sim/get_piano_dimensions.py
"""

import argparse
from isaaclab.app import AppLauncher

# Create argparser
parser = argparse.ArgumentParser(description="Extract piano dimensions from USD file.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now we have access to USD modules
from pxr import Usd, UsdGeom, Gf
import isaaclab.sim as sim_utils
import isaaclab.sim.utils.prims as prim_utils

def get_prim_bounds(prim_path: str, stage: Usd.Stage):
    """Get the bounding box of a USD prim."""
    prim = stage.GetPrimAtPath(prim_path)
    
    if not prim.IsValid():
        print(f"[ERROR]: Prim at {prim_path} is not valid")
        return None
    
    if not prim.IsA(UsdGeom.Boundable):
        print(f"[WARNING]: Prim at {prim_path} is not boundable")
        return None
    
    # Compute bounding box
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    bbox = bbox_cache.ComputeWorldBound(prim)
    
    if not bbox:
        print(f"[WARNING]: Could not compute bounds for {prim_path}")
        return None
    
    bounds = bbox.ComputeAlignedRange()
    min_pt = bounds.GetMin()
    max_pt = bounds.GetMax()
    
    return {
        'min': (min_pt[0], min_pt[1], min_pt[2]),
        'max': (max_pt[0], max_pt[1], max_pt[2]),
        'size': (max_pt[0] - min_pt[0], max_pt[1] - min_pt[1], max_pt[2] - min_pt[2]),
        'center': ((min_pt[0] + max_pt[0]) / 2, (min_pt[1] + max_pt[1]) / 2, (min_pt[2] + max_pt[2]) / 2)
    }


def print_prim_hierarchy(prim, indent=0):
    """Recursively print prim hierarchy."""
    prefix = "  " * indent
    type_name = prim.GetTypeName()
    print(f"{prefix}{prim.GetPath()} ({type_name})")
    
    for child in prim.GetChildren():
        print_prim_hierarchy(child, indent + 1)


def main():
    """Main function to analyze piano dimensions."""
    
    # Initialize simulation context (needed for USD access)
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Load the piano USD as a prim
    piano_usd_path = "/home/solotech007/RoboGym/simulation/g1-piano-play/onshape-assets/piano/piano/piano.usd"
    
    print("\n" + "=" * 80)
    print("PIANO USD ANALYSIS")
    print("=" * 80)
    
    # Spawn the piano in the scene
    cfg_piano = sim_utils.UsdFileCfg(usd_path=piano_usd_path)
    cfg_piano.func("/World/Piano", cfg_piano, translation=(0.0, 0.0, 0.0))
    
    # Reset to initialize
    sim.reset()
    
    # Get the current stage
    stage = prim_utils.get_current_stage()
    
    # Print hierarchy
    print("\n[INFO]: Piano Prim Hierarchy:")
    print("-" * 80)
    piano_prim = stage.GetPrimAtPath("/World/Piano")
    if piano_prim.IsValid():
        print_prim_hierarchy(piano_prim)
    else:
        print("[ERROR]: Piano prim not found at /World/Piano")
        return
    
    # Get bounds of the entire piano
    print("\n[INFO]: Piano Bounding Box:")
    print("-" * 80)
    bounds = get_prim_bounds("/World/Piano", stage)
    
    if bounds:
        print(f"Min point: ({bounds['min'][0]:.4f}, {bounds['min'][1]:.4f}, {bounds['min'][2]:.4f})")
        print(f"Max point: ({bounds['max'][0]:.4f}, {bounds['max'][1]:.4f}, {bounds['max'][2]:.4f})")
        print(f"Size:      ({bounds['size'][0]:.4f}, {bounds['size'][1]:.4f}, {bounds['size'][2]:.4f})")
        print(f"Center:    ({bounds['center'][0]:.4f}, {bounds['center'][1]:.4f}, {bounds['center'][2]:.4f})")
        
        # Estimate keyboard area (typically front section of piano)
        print("\n[INFO]: Estimated Keyboard Area:")
        print("-" * 80)
        print("Assuming keyboard is along X-axis (width):")
        print(f"  Keyboard width: {bounds['size'][0]:.4f}m")
        print(f"  Left edge X:  {bounds['min'][0]:.4f}m")
        print(f"  Right edge X: {bounds['max'][0]:.4f}m")
        print(f"  Center X:     {bounds['center'][0]:.4f}m")
        print(f"  Keyboard Y (front): {bounds['min'][1]:.4f}m")
        print(f"  Key height Z: {bounds['max'][2]:.4f}m")
        
        # Calculate suggested hand positions
        print("\n[INFO]: Suggested Hand Target Positions:")
        print("-" * 80)
        keyboard_y = bounds['min'][1]  # Front of piano
        keyboard_z = bounds['max'][2] + 0.05  # Slightly above keys
        left_x = bounds['min'][0] + 0.1  # 10cm from left edge
        right_x = bounds['max'][0] - 0.1  # 10cm from right edge
        center_x = bounds['center'][0]
        
        print(f"Left hand start:  ({left_x:.3f}, {keyboard_y:.3f}, {keyboard_z:.3f})")
        print(f"Right hand start: ({right_x:.3f}, {keyboard_y:.3f}, {keyboard_z:.3f})")
        print(f"Center position:  ({center_x:.3f}, {keyboard_y:.3f}, {keyboard_z:.3f})")
        
        # Export as Python dict for easy use
        print("\n[INFO]: Python Dictionary (copy to your script):")
        print("-" * 80)
        print("piano_keyboard = {")
        print(f"    'width': {bounds['size'][0]:.4f},")
        print(f"    'left_edge_x': {bounds['min'][0]:.4f},")
        print(f"    'right_edge_x': {bounds['max'][0]:.4f},")
        print(f"    'center_x': {bounds['center'][0]:.4f},")
        print(f"    'front_y': {bounds['min'][1]:.4f},")
        print(f"    'top_z': {bounds['max'][2]:.4f},")
        print(f"    'hand_height_z': {keyboard_z:.4f},")
        print(f"    'left_hand_start': ({left_x:.3f}, {keyboard_y:.3f}, {keyboard_z:.3f}),")
        print(f"    'right_hand_start': ({right_x:.3f}, {keyboard_y:.3f}, {keyboard_z:.3f}),")
        print(f"    'center_pos': ({center_x:.3f}, {keyboard_y:.3f}, {keyboard_z:.3f}),")
        print("}")
    else:
        print("[ERROR]: Could not compute piano bounds")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()

