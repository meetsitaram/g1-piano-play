#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test script for G1 Piano Reaching environment.

This script tests that the environment can be created and run with random actions.
Use this to verify setup before training.

Usage:
    # Test with visualization
    ./isaaclab.sh -p g1-piano-play/scripts/test_env.py
    
    # Test headless (faster)
    ./isaaclab.sh -p g1-piano-play/scripts/test_env.py --headless
    
    # Test with specific number of envs
    ./isaaclab.sh -p g1-piano-play/scripts/test_env.py --num_envs 16
"""

import argparse
import torch

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test G1 Piano Reaching environment")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to test.")
parser.add_argument("--task", type=str, default="Isaac-Piano-Reach-G1-v0", help="Task to test.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps to run.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import sys
import os

# Add parent directory to path to import envs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs import G1PianoReachEnv, G1PianoReachEnvCfg


def main():
    """Test the environment."""
    
    print("\n" + "=" * 80)
    print("[TEST]: G1 Piano Reaching Environment")
    print("=" * 80)
    
    # Create environment configuration
    env_cfg = G1PianoReachEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    print(f"\n[INFO]: Creating environment with {env_cfg.scene.num_envs} parallel environments...")
    
    try:
        # Create environment
        env = G1PianoReachEnv(cfg=env_cfg)
        
        print("\n[SUCCESS]: Environment created successfully!")
        print(f"  - Observation space: {env.observation_space}")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Number of environments: {env.num_envs}")
        print(f"  - Device: {env.device}")
        print(f"  - Action dimension (from config): {env.unwrapped.cfg.action_space}")
        print(f"  - Observation dimension (from config): {env.unwrapped.cfg.observation_space}")
        
        # Reset environment
        print("\n[INFO]: Resetting environment...")
        obs, info = env.reset()
        
        print(f"[SUCCESS]: Environment reset successfully!")
        print(f"  - Observation shape: {obs['policy'].shape}")
        print(f"  - Observation range: [{obs['policy'].min():.3f}, {obs['policy'].max():.3f}]")
        
        # Run random actions
        print(f"\n[INFO]: Running {args_cli.num_steps} steps with random actions...")
        
        episode_rewards = torch.zeros(env.num_envs, device=env.device)
        episode_lengths = torch.zeros(env.num_envs, device=env.device)
        num_resets = 0
        
        for step in range(args_cli.num_steps):
            # Sample random actions
            # Get correct action dimension from config (10 DOF for arms)
            action_dim = env.unwrapped.cfg.action_space if hasattr(env.unwrapped.cfg, 'action_space') else 10
            actions = torch.randn(env.num_envs, action_dim, device=env.device)
            actions = torch.clamp(actions, -1.0, 1.0)  # Clip to [-1, 1]
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Track episode statistics
            episode_rewards += rewards
            episode_lengths += 1
            
            # Check for resets
            resets = terminated | truncated
            if resets.any():
                reset_ids = resets.nonzero(as_tuple=False).flatten()
                num_resets += len(reset_ids)
                
                # Print episode stats for reset envs
                for env_id in reset_ids:
                    print(f"  [Env {env_id.item()}] Episode finished: "
                          f"reward={episode_rewards[env_id].item():.2f}, "
                          f"length={episode_lengths[env_id].item():.0f}")
                
                # Reset tracking for these envs
                episode_rewards[reset_ids] = 0.0
                episode_lengths[reset_ids] = 0.0
            
            # Print progress every 100 steps
            if (step + 1) % 100 == 0:
                print(f"  Step {step + 1}/{args_cli.num_steps} - "
                      f"Avg reward: {rewards.mean():.3f}, "
                      f"Resets: {num_resets}")
        
        print(f"\n[SUCCESS]: Completed {args_cli.num_steps} steps!")
        print(f"  - Total resets: {num_resets}")
        print(f"  - Average episode length: {args_cli.num_steps / max(num_resets, 1):.1f}")
        
        # Test specific observations
        print("\n[INFO]: Checking observations...")
        obs_dim = obs['policy'].shape[-1]
        print(f"  - Total observation dimension: {obs_dim}")
        if obs_dim >= 36:
            print(f"  - Arm joint positions (first 10 dims): {obs['policy'][0, :10]}")
            print(f"  - Arm joint velocities (next 10 dims): {obs['policy'][0, 10:20]}")
            print(f"  - Left hand to target (next 3 dims): {obs['policy'][0, 20:23]}")
            print(f"  - Right hand to target (next 3 dims): {obs['policy'][0, 23:26]}")
            print(f"  - Previous actions (last 10 dims): {obs['policy'][0, 26:36]}")
        else:
            print(f"  - Observation vector: {obs['policy'][0]}")
        
        # Close environment
        env.close()
        
        print("\n" + "=" * 80)
        print("[TEST PASSED]: Environment is working correctly! ✓")
        print("=" * 80)
        print("\nYou can now proceed to training:")
        print(f"  ./isaaclab.sh -p g1-piano-play/scripts/train.py --task {args_cli.task} --num_envs 1024")
        print()
        
    except Exception as e:
        print(f"\n[ERROR]: Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 80)
        print("[TEST FAILED]: Please fix the errors above before training. ✗")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

