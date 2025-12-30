#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Training script for G1 Piano Reaching task using RL Games.

Usage:
    # Basic training
    ./isaaclab.sh -p g1-piano-play/scripts/train.py --task Isaac-Piano-Reach-G1-v0
    
    # With custom settings
    ./isaaclab.sh -p g1-piano-play/scripts/train.py \\
        --task Isaac-Piano-Reach-G1-v0 \\
        --num_envs 1024 \\
        --headless
    
    # Resume from checkpoint
    ./isaaclab.sh -p g1-piano-play/scripts/train.py \\
        --task Isaac-Piano-Reach-G1-v0 \\
        --checkpoint logs/rl_games/Isaac-Piano-Reach-G1-v0/nn/G1PianoReach_ep_100.pth
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train G1 to reach piano with RL Games PPO")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Piano-Reach-G1-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras for video recording
if args_cli.video:
    args_cli.enable_cameras = True

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
from datetime import datetime

from omegaconf import OmegaConf

from isaaclab.envs import (
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

# Import custom environment to register it
# Add parent directory to path so g1-piano-play can be imported as g1_piano_play
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import envs  # noqa: F401 - registers G1PianoReachEnv

# Import rl_games
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner


def main():
    """Train with RL-Games PPO."""
    
    # Parse configuration
    env_cfg: DirectRLEnvCfg | ManagerBasedRLEnvCfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    
    # Override default num_envs if specified
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    
    # Create Isaac Gym environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("videos", "train", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO]: Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Set up RL-Games configuration
    # Use script directory as base for finding config file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..")
    rl_games_cfg_path = os.path.join(project_root, "envs", "agents", "rl_games_ppo_cfg.yaml")
    
    # Read and modify RL-Games config
    rl_games_cfg_omega = OmegaConf.load(rl_games_cfg_path)
    rl_games_cfg_omega.params.config.name = args_cli.task
    rl_games_cfg_omega.params.config.num_actors = env.unwrapped.num_envs
    if args_cli.seed is not None:
        rl_games_cfg_omega.params.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        rl_games_cfg_omega.params.config.max_epochs = args_cli.max_iterations
    
    # Handle checkpoint loading for resuming training
    if args_cli.checkpoint is not None:
        rl_games_cfg_omega.params.load_checkpoint = True
        rl_games_cfg_omega.params.load_path = args_cli.checkpoint
        print(f"[INFO]: Loading model checkpoint from: {args_cli.checkpoint}")
    
    # Extract wrapper parameters from config
    rl_device = rl_games_cfg_omega.params.config.device
    clip_obs = rl_games_cfg_omega.params.env.get("clip_observations", math.inf)
    clip_actions = rl_games_cfg_omega.params.env.get("clip_actions", math.inf)
    obs_groups = rl_games_cfg_omega.params.env.get("obs_groups", None)
    concate_obs_groups = rl_games_cfg_omega.params.env.get("concate_obs_groups", True)
    
    # Convert OmegaConf to dict for RL Games (it will modify the config internally)
    rl_games_cfg = OmegaConf.to_container(rl_games_cfg_omega, resolve=True)
    
    # Wrap for RL-Games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    
    # Register environment
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    
    # Specify directory for logging runs
    log_root_path = os.path.join("logs", "rl_games", args_cli.task)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO]: Logging runs to: {log_root_path}")
    
    # Create runner
    runner = Runner(IsaacAlgoObserver())
    runner.load(rl_games_cfg)
    runner.reset()
    
    # Run training
    runner.run({
        "train": True,
        "play": False,
    })
    
    # Close environment
    env.close()


if __name__ == "__main__":
    # Run training
    try:
        main()
    except Exception as e:
        print(f"[ERROR]: Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close simulation app
        simulation_app.close()

