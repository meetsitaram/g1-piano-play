#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test trained policy visually with shorter episodes for rapid cycling.

Usage:
    ./isaaclab.sh -p g1-piano-play/scripts/test_policy_visual.py \
        --checkpoint runs/Isaac-Piano-Reach-G1-v0_30-09-59-42/nn/last_Isaac-Piano-Reach-G1-v0_ep_500_rew_4753.192.pth \
        --num_envs 4 \
        --episode_length 150 \
        --num_episodes 20
"""

import argparse
import os
import sys
import torch

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test trained G1 Piano policy with custom episode length")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Piano-Reach-G1-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth file).")
parser.add_argument("--num_episodes", type=int, default=20, help="Number of episodes to run per environment.")
parser.add_argument("--episode_length", type=int, default=150, help="Maximum episode length (steps).")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
from omegaconf import OmegaConf

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

# Import custom environment
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import envs  # noqa: F401


def main():
    """Test trained policy with custom episode length."""
    
    # Parse configuration
    env_cfg: DirectRLEnvCfg | ManagerBasedRLEnvCfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    
    # Override num_envs and episode length
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = args_cli.episode_length * env_cfg.sim.dt  # Convert steps to seconds
    
    print(f"[INFO]: Episode length set to: {args_cli.episode_length} steps ({env_cfg.episode_length_s:.2f}s)")
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    
    # Load RL Games config for wrapper params
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..")
    rl_games_cfg_path = os.path.join(project_root, "envs", "agents", "rl_games_ppo_cfg.yaml")
    rl_games_cfg = OmegaConf.load(rl_games_cfg_path)
    
    # Extract wrapper parameters
    rl_device = args_cli.device
    clip_obs = rl_games_cfg.params.env.get("clip_observations", math.inf)
    clip_actions = rl_games_cfg.params.env.get("clip_actions", math.inf)
    
    # Wrap environment
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, None, True)
    
    # Load checkpoint
    print(f"[INFO]: Loading policy from: {args_cli.checkpoint}")
    if not os.path.exists(args_cli.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args_cli.checkpoint}")
    
    checkpoint = torch.load(args_cli.checkpoint, map_location=rl_device, weights_only=False)
    
    # Use RL Games Runner to load policy
    from rl_games.torch_runner import Runner
    from rl_games.common import env_configurations, vecenv
    from isaaclab_rl.rl_games import RlGamesGpuEnv
    
    # Load full config
    rl_games_cfg_full = OmegaConf.to_container(rl_games_cfg, resolve=True)
    rl_games_cfg_full['params']['config']['name'] = args_cli.task
    rl_games_cfg_full['params']['config']['num_actors'] = env.num_envs
    rl_games_cfg_full['params']['load_checkpoint'] = True
    rl_games_cfg_full['params']['load_path'] = args_cli.checkpoint
    
    # Register environment
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    
    # Create runner and load checkpoint
    runner = Runner()
    runner.load(rl_games_cfg_full)
    
    # Get the player (inference agent)
    agent = runner.create_player()
    agent.restore(args_cli.checkpoint)
    agent.reset()
    
    print("[INFO]: Policy loaded successfully!")
    print(f"[INFO]: Testing for {args_cli.num_episodes} episodes with {args_cli.episode_length} steps each")
    print(f"[INFO]: Using {args_cli.num_envs} parallel environments")
    print("="*80)
    
    # Run inference
    obs_raw = env.reset()
    
    print(f"[DEBUG] Reset returned type: {type(obs_raw)}")
    if isinstance(obs_raw, dict):
        print(f"[DEBUG] Reset dict keys: {obs_raw.keys()}")
        for k, v in obs_raw.items():
            if isinstance(v, dict):
                print(f"[DEBUG]   {k}: dict with keys={v.keys()}")
            else:
                print(f"[DEBUG]   {k}: type={type(v)}, shape={v.shape if hasattr(v, 'shape') else 'N/A'}")
    
    # Store the raw observation (will be extracted in loop)
    obs = obs_raw
    
    # Extract initial obs tensor for batch size setup
    if isinstance(obs, dict):
        initial_obs_tensor = obs["obs"] if "obs" in obs else list(obs.values())[0]
    else:
        initial_obs_tensor = obs
    
    # Statistics tracking
    episode_count = torch.zeros(env.num_envs, device=rl_device, dtype=torch.int32)
    episode_rewards = torch.zeros(env.num_envs, device=rl_device)
    episode_lengths = torch.zeros(env.num_envs, device=rl_device)
    
    completed_episodes = 0
    total_reward = 0.0
    total_length = 0
    
    # Enable batch mode - pass the tensor, not the dict
    _ = agent.get_batch_size(initial_obs_tensor, 1)
    
    # Initialize RNN if used
    if agent.is_rnn:
        agent.init_rnn()
    
    print("\n🎹 WATCHING TRAINED POLICY PLAY PIANO 🤖\n")
    
    step_num = 0
    while completed_episodes < args_cli.num_episodes * args_cli.num_envs:
        step_num += 1
        
        # Get action from trained policy
        
        # Extract tensor from observation (env wrapper returns {'obs': tensor})
        if isinstance(obs, dict):
            # Try standard keys
            if "obs" in obs:
                obs_tensor = obs["obs"]
            elif "policy" in obs:
                obs_tensor = obs["policy"]
            elif len(obs) == 1:
                obs_tensor = list(obs.values())[0]
            else:
                raise RuntimeError(f"Cannot extract tensor from obs dict with keys: {obs.keys()}")
        else:
            obs_tensor = obs
        
        # Debug: check what we extracted
        if step_num == 1:
            print(f"[DEBUG] obs type: {type(obs)}")
            if isinstance(obs, dict):
                print(f"[DEBUG] obs keys: {obs.keys()}")
            print(f"[DEBUG] obs_tensor type: {type(obs_tensor)}")
            if isinstance(obs_tensor, dict):
                print(f"[DEBUG] ERROR: obs_tensor is still a dict! Keys: {obs_tensor.keys()}")
            else:
                print(f"[DEBUG] obs_tensor shape: {obs_tensor.shape}")
        
        # Verify it's a tensor
        if not isinstance(obs_tensor, torch.Tensor):
            raise RuntimeError(f"Observation must be a tensor! Got: {type(obs_tensor)}, value: {obs_tensor}")
        
        if step_num == 1:
            print(f"[DEBUG] Before obs_to_torch: type={type(obs_tensor)}, shape={obs_tensor.shape}")
        
        # Convert observation to agent format (critical step!)
        obs_agent = agent.obs_to_torch(obs_tensor)
        
        if step_num == 1:
            print(f"[DEBUG] After obs_to_torch: type={type(obs_agent)}")
        
        # Get action from policy
        action = agent.get_action(obs_agent, is_deterministic=True)
        
        # Step environment
        obs_result, rewards, dones, infos = env.step(action)
        
        # Store the raw result for next iteration (don't extract yet)
        obs = obs_result
        
        # Update statistics
        episode_rewards += rewards
        episode_lengths += 1
        
        # Check for done episodes
        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_indices) > 0:
            for idx in done_indices:
                episode_count[idx] += 1
                completed_episodes += 1
                
                reward = episode_rewards[idx].item()
                length = episode_lengths[idx].item()
                
                total_reward += reward
                total_length += length
                
                # Visual indicator of performance
                if reward > 3000:
                    status = "🎹✨ EXCELLENT!"
                elif reward > 1000:
                    status = "✅ GOOD"
                elif reward > 0:
                    status = "👍 OK"
                else:
                    status = "⚠️  LEARNING"
                
                print(f"[Env {idx+1} | Episode {episode_count[idx]:3d}] Reward: {reward:8.2f} | Length: {length:3.0f} | {status}")
                
                # Reset stats for this env
                episode_rewards[idx] = 0.0
                episode_lengths[idx] = 0
    
    # Print summary
    total_episodes = args_cli.num_episodes * args_cli.num_envs
    avg_reward = total_reward / total_episodes
    avg_length = total_length / total_episodes
    
    print("\n" + "="*80)
    print(f"📊 RESULTS ({total_episodes} total episodes)")
    print("="*80)
    print(f"Average Reward:  {avg_reward:8.2f}")
    print(f"Average Length:  {avg_length:8.2f} steps")
    print("="*80)
    
    # Close environment
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR]: Testing failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()

