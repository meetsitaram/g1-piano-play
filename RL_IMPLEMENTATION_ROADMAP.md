# RL Implementation Roadmap - Quick Start Guide

## Day-by-Day Implementation Plan

### Day 1: Project Structure & Environment Skeleton

**Goal**: Create file structure and basic environment shell

**Tasks**:
1. Create directory structure
   ```bash
   cd /home/solotech007/RoboGym/simulation/g1-piano-play
   mkdir -p envs/agents scripts
   touch envs/__init__.py
   touch envs/g1_piano_reach_env_cfg.py
   touch envs/g1_piano_reach_env.py
   touch envs/agents/__init__.py
   ```

2. Copy cartpole as template
   ```bash
   # Use as reference
   cp ../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/cartpole_env.py envs/g1_piano_reach_env.py
   ```

3. Create minimal config class
   ```python
   # In g1_piano_reach_env_cfg.py
   @configclass
   class G1PianoReachEnvCfg(DirectRLEnvCfg):
       decimation = 2
       episode_length_s = 10.0
       action_space = 10  # 5 DOF per arm
       observation_space = 36
       # ... (see full plan)
   ```

**Deliverable**: File structure exists, imports work

**Time**: 2-3 hours

---

### Day 2: Scene Setup (Reuse Existing Code)

**Goal**: Adapt `g1-piano-play.py` scene into RL environment

**Tasks**:
1. Copy `design_scene()` logic to `_setup_scene()` method
   ```python
   def _setup_scene(self):
       # Spawn G1 robot as Articulation
       self.robot = Articulation(self.cfg.robot_cfg)
       
       # Spawn bench (copy from g1-piano-play.py line 169-182)
       # Spawn table (copy from g1-piano-play.py line 116-131)
       # Spawn piano (copy from g1-piano-play.py line 149-155)
       # Add ground, lights, etc.
       
       # Clone environments (NEW - for parallelization)
       self.scene.clone_environments(copy_from_source=False)
       
       # Add robot to scene
       self.scene.articulations["robot"] = self.robot
   ```

2. Test with single environment
   ```python
   # Test script
   from envs.g1_piano_reach_env_cfg import G1PianoReachEnvCfg
   from envs.g1_piano_reach_env import G1PianoReachEnv
   
   cfg = G1PianoReachEnvCfg()
   cfg.scene.num_envs = 1  # Test with 1 env first
   env = G1PianoReachEnv(cfg)
   ```

3. Debug scene creation
   - Check all assets spawn
   - Verify robot is on bench
   - Confirm piano is in front

**Deliverable**: Scene spawns successfully with 1-16 parallel environments

**Time**: 3-4 hours

---

### Day 3: Joint Indexing & Hand Positions

**Goal**: Get arm joint indices and hand body positions

**Tasks**:
1. Find arm joints
   ```python
   def _get_arm_joint_indices(self):
       # Print all joint names first
       print("All joints:", self.robot.data.joint_names)
       
       # Find arm joints (shoulder and elbow)
       arm_joints = [
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
       for joint_name in arm_joints:
           idx = self.robot.data.joint_names.index(joint_name)
           indices.append(idx)
       
       return torch.tensor(indices, device=self.device)
   ```

2. Find hand body indices
   ```python
   def _find_hand_bodies(self):
       # Print all body names
       print("All bodies:", self.robot.body_names)
       
       # Find hands (might be "left_palm_link", "left_hand_link", etc.)
       # Check actual names from print output
       left_hand = self.robot.find_bodies("left_.*palm.*")[0]  # Regex
       right_hand = self.robot.find_bodies("right_.*palm.*")[0]
       
       return left_hand[0], right_hand[0]  # Return indices
   ```

3. Test position extraction
   ```python
   # In a test loop
   left_hand_pos = self.robot.data.body_pos_w[:, self._left_hand_idx]
   print(f"Left hand position: {left_hand_pos[0]}")  # Print first env
   ```

**Deliverable**: Can access arm joint states and hand positions

**Time**: 2-3 hours

---

### Day 4: Observation & Action Implementation

**Goal**: Implement `_get_observations()` and `_apply_action()`

**Tasks**:
1. Implement `_get_observations()`
   ```python
   def _get_observations(self) -> dict:
       # Get arm joint states
       arm_pos = self.robot.data.joint_pos[:, self._arm_joint_indices]
       arm_vel = self.robot.data.joint_vel[:, self._arm_joint_indices]
       
       # Get hand positions
       left_hand_pos = self.robot.data.body_pos_w[:, self._left_hand_idx]
       right_hand_pos = self.robot.data.body_pos_w[:, self._right_hand_idx]
       
       # Compute relative to targets
       left_to_target = left_hand_pos - self.left_target_w
       right_to_target = right_hand_pos - self.right_target_w
       
       # Concatenate
       obs = torch.cat([
           arm_pos,           # 10
           arm_vel,           # 10
           left_to_target,    # 3
           right_to_target,   # 3
           self.previous_actions,  # 10
       ], dim=-1)
       
       return {"policy": obs}
   ```

2. Implement action application
   ```python
   def _apply_action(self):
       # Apply to arms
       self.robot.set_joint_position_target(
           self.actions, 
           joint_ids=self._arm_joint_indices
       )
       
       # Keep legs in sitting pose
       self.robot.set_joint_position_target(
           self._sitting_pose,
           joint_ids=self._leg_joint_indices
       )
   ```

3. Test observation shape
   ```python
   obs = env.reset()
   print("Obs shape:", obs["policy"].shape)  # Should be (num_envs, 36)
   ```

**Deliverable**: Observations and actions work, shapes are correct

**Time**: 3-4 hours

---

### Day 5: Reward Function (Simple Version)

**Goal**: Basic distance-based reward

**Tasks**:
1. Implement simple distance reward
   ```python
   def _get_rewards(self) -> torch.Tensor:
       # Get hand positions
       left_hand = self.robot.data.body_pos_w[:, self._left_hand_idx]
       right_hand = self.robot.data.body_pos_w[:, self._right_hand_idx]
       
       # Distance to targets
       left_dist = torch.norm(left_hand - self.left_target_w, dim=-1)
       right_dist = torch.norm(right_hand - self.right_target_w, dim=-1)
       
       # Simple negative distance reward
       reward = -1.0 * (left_dist + right_dist)
       
       return reward
   ```

2. Test reward values
   ```python
   # Run a few steps and print rewards
   for i in range(10):
       obs, reward, done, truncated, info = env.step(random_action)
       print(f"Step {i}: reward = {reward[0].item():.3f}")
   ```

**Deliverable**: Rewards computed, values are reasonable

**Time**: 2 hours

---

### Day 6: Reset & Termination Logic

**Goal**: Implement `_reset_idx()` and `_get_dones()`

**Tasks**:
1. Implement reset
   ```python
   def _reset_idx(self, env_ids):
       if env_ids is None:
           env_ids = self.robot._ALL_INDICES
       super()._reset_idx(env_ids)
       
       # Reset to sitting pose + random arm positions
       joint_pos = self._default_joint_pos[env_ids].clone()
       joint_pos[:, self._arm_joint_indices] += torch.randn(...) * 0.1
       
       # Write to sim
       self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
   ```

2. Implement termination
   ```python
   def _get_dones(self):
       time_out = self.episode_length_buf >= self.max_episode_length - 1
       fell = self.robot.data.root_pos_w[:, 2] < 0.3
       return fell, time_out
   ```

**Deliverable**: Environment can reset and terminate properly

**Time**: 2-3 hours

---

### Day 7: Gym Registration & Test

**Goal**: Register environment and test full loop

**Tasks**:
1. Register with Gymnasium
   ```python
   # In envs/__init__.py
   import gymnasium as gym
   from .g1_piano_reach_env import G1PianoReachEnv
   from .g1_piano_reach_env_cfg import G1PianoReachEnvCfg
   
   gym.register(
       id="Isaac-Piano-Reach-G1-v0",
       entry_point="g1-piano-play.envs:G1PianoReachEnv",
       kwargs={"cfg": G1PianoReachEnvCfg},
   )
   ```

2. Create test script
   ```python
   # scripts/test_env.py
   import gymnasium as gym
   
   env = gym.make("Isaac-Piano-Reach-G1-v0", num_envs=16)
   
   obs, info = env.reset()
   print("Initial obs:", obs["policy"].shape)
   
   for i in range(100):
       action = torch.randn(16, 10)  # Random actions
       obs, reward, terminated, truncated, info = env.step(action)
       
       if i % 10 == 0:
           print(f"Step {i}: reward={reward.mean():.3f}")
   
   env.close()
   ```

3. Run full test
   ```bash
   ./isaaclab.sh -p g1-piano-play/scripts/test_env.py
   ```

**Deliverable**: Environment runs end-to-end without errors

**Time**: 2-3 hours

---

### Week 2: Training Integration

**Day 8-9: RL Games Integration**

1. Create training config (copy from cartpole)
   ```yaml
   # envs/agents/rl_games_ppo_cfg.yaml
   # See full config in main plan
   ```

2. Create training script
   ```python
   # scripts/train.py
   from rl_games.torch_runner import Runner
   # ... (see examples)
   ```

3. First training run (16 envs, 100 epochs)
   ```bash
   ./isaaclab.sh -p g1-piano-play/scripts/train.py --num_envs 16 --max_epochs 100
   ```

**Day 10-11: Debug Training**

1. Check for NaN/Inf in losses
2. Visualize learning curves
3. Adjust learning rate if needed
4. Test with different reward scales

**Day 12-14: Scale Up**

1. Increase to 256 envs
2. Train for 500 epochs
3. Monitor success metrics
4. Save best checkpoint

---

### Week 3: Reward Tuning & Polish

**Day 15-17: Reward Engineering**

1. Add contact sensing (if simple distance isn't enough)
2. Add action smoothness penalty
3. Add joint limit penalty
4. Test different reward combinations

**Day 18-19: Hyperparameter Tuning**

1. Learning rate sweep
2. Network size experiments
3. Horizon length tuning
4. Entropy coefficient tuning

**Day 20-21: Evaluation**

1. Test final policy
2. Record videos
3. Measure success rate
4. Document results

---

## Quick Reference: Common Issues

### Issue: "Can't find body 'left_palm_link'"
**Solution**: Print all body names first
```python
print("All bodies:", robot.body_names)
# Look for: "left_hand", "l_palm", "left_gripper", etc.
```

### Issue: Observation shape mismatch
**Solution**: Check each component shape
```python
print("Arm pos:", arm_pos.shape)
print("Arm vel:", arm_vel.shape)
print("Left to target:", left_to_target.shape)
# Make sure they all match expected dimensions
```

### Issue: Reward is always negative
**Solution**: Check target positions are set correctly
```python
print("Left target:", self.left_target_w[0])
print("Left hand:", left_hand_pos[0])
print("Distance:", torch.norm(left_hand_pos[0] - self.left_target_w[0]))
```

### Issue: Robot falls immediately
**Solution**: Check sitting pose is applied correctly
```python
# Make sure legs are set to sitting positions
# Check in _apply_action() that leg targets are set
```

### Issue: Training doesn't improve
**Checklist**:
- [ ] Rewards are reasonable magnitude (-10 to +10 range)
- [ ] Observations are normalized (or at least bounded)
- [ ] Learning rate is appropriate (try 3e-4)
- [ ] Network size is sufficient (256, 128, 64)
- [ ] Episode length is long enough (>5 seconds)
- [ ] Enough environments (>256)

---

## Minimal Working Example

Here's the absolute minimum to get started:

```python
# g1_piano_reach_env_cfg.py
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass

@configclass
class G1PianoReachEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 10.0
    action_space = 10
    observation_space = 36
    # ... rest of config

# g1_piano_reach_env.py
from isaaclab.envs import DirectRLEnv

class G1PianoReachEnv(DirectRLEnv):
    cfg: G1PianoReachEnvCfg
    
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # Cache indices
        self._setup_indices()
    
    def _setup_scene(self):
        # Spawn robot, piano, bench, table
        # (Copy from g1-piano-play.py)
        pass
    
    def _get_observations(self) -> dict:
        # Return {"policy": obs_tensor}
        pass
    
    def _apply_action(self):
        # Set joint targets
        pass
    
    def _get_rewards(self) -> torch.Tensor:
        # Compute rewards
        pass
    
    def _get_dones(self) -> tuple:
        # Return (terminated, truncated)
        pass
    
    def _reset_idx(self, env_ids):
        # Reset environments
        pass
```

---

## Success Checklist

Use this to track progress:

### Week 1
- [ ] File structure created
- [ ] Scene spawns with 16 envs
- [ ] Can access arm joints
- [ ] Can access hand positions
- [ ] Observations computed
- [ ] Actions applied
- [ ] Rewards computed
- [ ] Reset works
- [ ] Termination works
- [ ] Gym registration works
- [ ] Full test runs without errors

### Week 2
- [ ] Training script runs
- [ ] Can train for 100 epochs
- [ ] Learning curve shows progress
- [ ] No NaN/Inf errors
- [ ] Can save checkpoints
- [ ] Can load and play policy
- [ ] Scaled to 256+ envs

### Week 3
- [ ] Hands move toward piano
- [ ] Contact detection works
- [ ] Success rate > 50%
- [ ] Can record videos
- [ ] Final policy saved
- [ ] Results documented

---

## Key References for Implementation

### Primary Examples to Study:
1. **Franka Cabinet** (`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/franka_cabinet/`) - Reaching task, CLOSEST MATCH
2. **Cartpole** (`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/`) - Simple structure, good baseline
3. **In-Hand Manipulation** (`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/inhand_manipulation/`) - Contact rewards
4. **Humanoid** (`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid/`) - Similar robot complexity

### Key Files to Reference:
- **Environment class**: `franka_cabinet_env.py` (primary reference)
- **Training config**: `franka_cabinet/agents/rl_games_ppo_cfg.yaml`
- **Base class**: `isaaclab/envs/direct_rl_env.py` (understand interface)
- **Articulation API**: `isaaclab/assets/articulation/articulation.py`

### Quick Commands:
```bash
# Copy franka as template
cp IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/franka_cabinet/franka_cabinet_env.py \
   g1-piano-play/envs/g1_piano_reach_env.py

# Copy training config
cp IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/franka_cabinet/agents/rl_games_ppo_cfg.yaml \
   g1-piano-play/envs/agents/rl_games_ppo_cfg.yaml

# Search for API usage
grep -r "body_pos_w" IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/
```

### Documentation Links:
- **Isaac Lab**: https://isaac-sim.github.io/IsaacLab/
- **RL Games**: https://github.com/Denys88/rl_games
- **RL Theory**: https://spinningup.openai.com/

**See [RL_PIANO_REACH_PLAN.md](./RL_PIANO_REACH_PLAN.md) → "Resources & References" section for complete detailed list**

---

## Next: Start Implementation!

**Recommended First Command:**
```bash
cd /home/solotech007/RoboGym/simulation/g1-piano-play
mkdir -p envs/agents scripts
touch envs/__init__.py
touch envs/g1_piano_reach_env_cfg.py
touch envs/g1_piano_reach_env.py
```

Then start with Day 1 tasks! 🚀

