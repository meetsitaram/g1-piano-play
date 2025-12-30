#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Play/inference script for trained G1 Piano Reaching policy.

Usage:
    # Play trained policy
    ./isaaclab.sh -p g1-piano-play/scripts/play.py \\
        --task Isaac-Piano-Reach-G1-v0 \\
        --checkpoint /path/to/model.pth
    
    # Play with video recording
    ./isaaclab.sh -p g1-piano-play/scripts/play.py \\
        --task Isaac-Piano-Reach-G1-v0 \\
        --checkpoint /path/to/model.pth \\
        --video
"""

import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Play trained G1 Piano Reaching policy")
parser.add_argument("--video", action="store_true", default=False, help="Record video of episodes.")
parser.add_argument("--video_length", type=int, default=500, help="Length of recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (usually 1 for playing).")
parser.add_argument("--task", type=str, default="Isaac-Piano-Reach-G1-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth file).")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Always enable cameras for video
if args_cli.video:
    args_cli.enable_cameras = True

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import os
import sys

# Add parent directory to path to import envs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import envs  # noqa: F401 - registers G1PianoReachEnv
from datetime import datetime

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Play trained policy."""
    
    # Parse configuration
    env_cfg: DirectRLEnvCfg | ManagerBasedRLEnvCfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    
    # Override num_envs
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("videos", "play", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
            "step_trigger": lambda step: step == 0,  # Record from start
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO]: Recording video during inference.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Load policy
    print(f"[INFO]: Loading policy from: {args_cli.checkpoint}")
    if not os.path.exists(args_cli.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args_cli.checkpoint}")
    
    policy = torch.jit.load(args_cli.checkpoint, map_location=args_cli.device)
    policy.eval()
    
    print("[INFO]: Policy loaded successfully!")
    
    # Run inference
    obs, _ = env.reset()
    episode_rewards = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    episode_lengths = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    num_episodes = 0
    
    print("\n[INFO]: Running policy inference...")
    
    with torch.inference_mode():
        while simulation_app.is_running():
            # Get action from policy
            action = policy(obs["policy"])
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(action)
            
            # Track statistics
            episode_rewards += rewards
            episode_lengths += 1
            
            # Check for episode completion
            resets = terminated | truncated
            if resets.any():
                reset_ids = resets.nonzero(as_tuple=False).flatten()
                
                for env_id in reset_ids:
                    num_episodes += 1
                    print(f"[Episode {num_episodes}] "
                          f"Reward: {episode_rewards[env_id].item():.2f}, "
                          f"Length: {episode_lengths[env_id].item():.0f}")
                    
                    # Reset tracking
                    episode_rewards[env_id] = 0.0
                    episode_lengths[env_id] = 0.0
    
    # Close environment
    env.close()
    
    print(f"\n[INFO]: Completed {num_episodes} episodes")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

