#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test a trained RL Games checkpoint for G1 Piano Reaching task.

Usage:
    # Test latest checkpoint with visualization
    ./isaaclab.sh -p g1-piano-play/scripts/test_checkpoint.py \\
        --task Isaac-Piano-Reach-G1-v0 \\
        --checkpoint logs/rl_games/Isaac-Piano-Reach-G1-v0/nn/last_checkpoint.pth \\
        --num_envs 4
    
    # Record video
    ./isaaclab.sh -p g1-piano-play/scripts/test_checkpoint.py \\
        --task Isaac-Piano-Reach-G1-v0 \\
        --checkpoint logs/rl_games/Isaac-Piano-Reach-G1-v0/nn/G1PianoReach_ep_500.pth \\
        --num_envs 16 \\
        --num_episodes 10
"""

import argparse
import os
import sys
import torch

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test trained G1 Piano Reaching policy")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Piano-Reach-G1-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth file).")
parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to run.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import numpy as np

from omegaconf import OmegaConf

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

# Import custom environment
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import envs  # noqa: F401


def main():
    """Test trained policy."""
    
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
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    
    # Load checkpoint config to get wrapper params
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
    
    # PyTorch 2.6+ requires weights_only=False for RL Games checkpoints (safe since we created them)
    checkpoint = torch.load(args_cli.checkpoint, map_location=rl_device, weights_only=False)
    
    # Extract model from checkpoint (RL Games checkpoint structure)
    if 'model' in checkpoint:
        model_state_dict = checkpoint['model']
    else:
        model_state_dict = checkpoint
    
    # Use RL Games Runner to properly load the checkpoint
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import torch_ext
    
    # Load the full RL Games config
    rl_games_cfg_full = OmegaConf.to_container(rl_games_cfg, resolve=True)
    rl_games_cfg_full['params']['config']['name'] = args_cli.task
    rl_games_cfg_full['params']['config']['num_actors'] = env.num_envs
    rl_games_cfg_full['params']['load_checkpoint'] = True
    rl_games_cfg_full['params']['load_path'] = args_cli.checkpoint
    
    # Register environment
    from rl_games.common import env_configurations, vecenv
    from isaaclab_rl.rl_games import RlGamesGpuEnv
    
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
    
    # Check if RNN
    if agent.is_rnn:
        print("[INFO]: Model uses RNN/LSTM")
    
    print("[INFO]: Policy loaded successfully!")
    print(f"[INFO]: Testing policy for {args_cli.num_episodes} episodes with {args_cli.num_envs} environments")
    
    # Run inference
    obs_dict_or_tensor = env.reset()
    
    # Extract observation tensor - be very explicit
    if isinstance(obs_dict_or_tensor, dict):
        if "obs" in obs_dict_or_tensor:
            obs = obs_dict_or_tensor["obs"]
        elif "policy" in obs_dict_or_tensor:
            obs = obs_dict_or_tensor["policy"]
        else:
            # Take first value if dict doesn't have expected keys
            obs = list(obs_dict_or_tensor.values())[0]
    else:
        obs = obs_dict_or_tensor
    
    # Ensure obs is a tensor
    if isinstance(obs, dict):
        raise RuntimeError(f"Failed to extract tensor from observation! Got: {type(obs)}, keys: {obs.keys()}")
    
    print(f"[DEBUG]: Observation shape: {obs.shape}, dtype: {obs.dtype}, device: {obs.device}")
    
    episode_count = torch.zeros(env.num_envs, device=rl_device, dtype=torch.int32)
    episode_rewards = torch.zeros(env.num_envs, device=rl_device)
    episode_lengths = torch.zeros(env.num_envs, device=rl_device)
    
    completed_episodes = 0
    total_reward = 0.0
    total_length = 0
    
    # Enable batch mode
    _ = agent.get_batch_size(obs, 1)
    
    # Initialize RNN if used
    if agent.is_rnn:
        agent.init_rnn()
    
    print("\n" + "="*80)
    print("TESTING POLICY")
    print("="*80)
    
    print("[WARNING]: Using FIXED 'arms straight forward' pose for testing")
    print("[INFO]: Arms should stretch forward toward piano")
    
    # Define EXTREME "arms straight forward" pose - make it VERY obvious
    # Joint order: left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow_pitch, left_elbow_roll,
    #              right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow_pitch, right_elbow_roll
    
    arms_forward_pose = torch.tensor([
        -1.5,  # left_shoulder_pitch: -1.5 rad (~-86 deg, VERY forward)
        0.5,   # left_shoulder_roll: outward
        0.0,   # left_shoulder_yaw: no rotation
        1.5,   # left_elbow_pitch: strong bend (within limits [0, 3.42])
        0.0,   # left_elbow_roll: straight
        -1.5,  # right_shoulder_pitch: -1.5 rad (~-86 deg, VERY forward)  
        -0.5,  # right_shoulder_roll: outward
        0.0,   # right_shoulder_yaw: no rotation
        1.5,   # right_elbow_pitch: strong bend (within limits [0, 3.42])
        0.0,   # right_elbow_roll: straight
    ], device=rl_device)
    
    print(f"\n[INFO]: Using EXTREME forward pose (3x stronger than before)")
    
    print(f"[DEBUG]: Arms forward pose: {arms_forward_pose}")
    print(f"[INFO]: Joint names: {env.unwrapped._arm_joint_names}")
    
    # Get joint limits for verification
    arm_joint_limits = env.unwrapped.robot.data.soft_joint_pos_limits[:, env.unwrapped._arm_joint_indices]
    print(f"[DEBUG]: Joint limits (first env):")
    for i, name in enumerate(env.unwrapped._arm_joint_names):
        lower = arm_joint_limits[0, i, 0].item()
        upper = arm_joint_limits[0, i, 1].item()
        target = arms_forward_pose[i].item()
        in_range = "✓" if lower <= target <= upper else "✗"
        print(f"  {name:30s}: [{lower:6.2f}, {upper:6.2f}] → target: {target:6.2f} {in_range}")
    
    step_count = 0
    print("\n[INFO]: Starting control loop...")
    print("[INFO]: Getting initial joint positions...\n")
    
    # Get initial joint positions
    initial_arm_pos = env.unwrapped.robot.data.joint_pos[0, env.unwrapped._arm_joint_indices].clone()  # Keep as tensor
    print(f"[DEBUG] Initial arm joint positions:")
    for i, (name, pos) in enumerate(zip(env.unwrapped._arm_joint_names, initial_arm_pos.cpu().numpy())):
        print(f"  {name:30s}: {pos:7.3f} rad")
    
    while completed_episodes < args_cli.num_episodes:
        # Get action from policy
        # Ensure obs is a tensor (sanity check)
        if not isinstance(obs, torch.Tensor):
            raise RuntimeError(f"Observation must be a tensor! Got: {type(obs)}")
        
        # Apply fixed "arms forward" pose
        action = arms_forward_pose.unsqueeze(0).expand(env.num_envs, -1)
        
        step_count += 1
        if step_count == 1:
            print(f"\n[INFO]: Applying arms forward pose... Watch the robot!")
            print(f"[INFO]: Target pose: {action[0]}")
        
        if step_count % 60 == 0 or step_count == 1:  # Every 60 steps (~1 second) and first step
            current_arm_pos = env.unwrapped.robot.data.joint_pos[0, env.unwrapped._arm_joint_indices]
            target_pos = action[0]
            
            # Calculate movement and accuracy
            movement = torch.sum(torch.abs(current_arm_pos - initial_arm_pos)).item()
            error = torch.sum(torch.abs(current_arm_pos - target_pos)).item()
            
            print(f"\n[DEBUG] Step {step_count} - Joint tracking:")
            print(f"  Left shoulder pitch:  current={current_arm_pos[0]:6.3f}, target={target_pos[0]:6.3f}, error={abs(current_arm_pos[0]-target_pos[0]):6.3f}")
            print(f"  Right shoulder pitch: current={current_arm_pos[5]:6.3f}, target={target_pos[5]:6.3f}, error={abs(current_arm_pos[5]-target_pos[5]):6.3f}")
            print(f"  Total movement from initial: {movement:.4f} rad")
            print(f"  Total error to target: {error:.4f} rad")
            
            if error < 0.5:
                print(f"  ✅ GOOD: Arms are tracking targets well!")
            elif movement > 0.5:
                print(f"  ⚠️  Arms are moving but not at target yet (converging...)")
            else:
                print(f"  ❌ WARNING: Arms not moving much!")
        
        # Step environment
        obs_result, rewards, dones, infos = env.step(action)
        
        # Extract observation tensor - be very explicit
        if isinstance(obs_result, dict):
            if "obs" in obs_result:
                obs = obs_result["obs"]
            elif "policy" in obs_result:
                obs = obs_result["policy"]
            else:
                # Take first value if dict doesn't have expected keys
                obs = list(obs_result.values())[0]
        else:
            obs = obs_result
        
        # Ensure obs is a tensor
        if isinstance(obs, dict):
            raise RuntimeError(f"Failed to extract tensor from step observation! Got: {type(obs)}, keys: {obs.keys()}")
        
        # Update statistics
        episode_rewards += rewards
        episode_lengths += 1
        
        # Check for done episodes
        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_indices) > 0:
            for idx in done_indices:
                if episode_count[idx] < args_cli.num_episodes:
                    episode_count[idx] += 1
                    completed_episodes += 1
                    
                    reward = episode_rewards[idx].item()
                    length = episode_lengths[idx].item()
                    
                    total_reward += reward
                    total_length += length
                    
                    print(f"[Episode {completed_episodes:3d}] Reward: {reward:8.2f} | Length: {length:4.0f} steps")
                    
                    # Reset stats for this env
                    episode_rewards[idx] = 0.0
                    episode_lengths[idx] = 0
                    
                    if completed_episodes >= args_cli.num_episodes:
                        break
    
    # Print summary
    avg_reward = total_reward / args_cli.num_episodes
    avg_length = total_length / args_cli.num_episodes
    
    print("="*80)
    print(f"RESULTS ({args_cli.num_episodes} episodes)")
    print("="*80)
    print(f"Average Reward:  {avg_reward:8.2f}")
    print(f"Average Length:  {avg_length:8.2f} steps")
    print("="*80)
    
    # Close environment
    env.close()


if __name__ == "__main__":
    # Run testing
    try:
        main()
    except Exception as e:
        print(f"[ERROR]: Testing failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()

