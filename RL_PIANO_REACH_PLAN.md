# G1 Piano Reaching - RL Training Plan

## Project Overview

**New Direction**: Instead of scripted animation with IK, we'll use **Reinforcement Learning** to train the G1 robot to reach toward and interact with a piano.

### Two-Phase Approach

**Phase 1: Simple Piano Touch** (2-3 weeks)
- G1 sits on bench (fixed sitting pose, not trainable)
- Only arm joints are controlled by RL
- Goal: Touch piano with both hands
- Rewards: Distance reduction + contact bonus

**Phase 2: Piano Key Pressing** (3-4 weeks)
- Replace simple piano with detailed keyboard model
- Each key is a separate target
- Input: Which key to press
- Goal: Press specific keys on command

---

## Implementation Architecture

### File Structure

```
g1-piano-play/
├── envs/
│   ├── __init__.py                          # Gym registration
│   ├── g1_piano_reach_env_cfg.py           # Phase 1: Environment configuration
│   ├── g1_piano_reach_env.py               # Phase 1: Environment implementation
│   ├── g1_piano_keys_env_cfg.py            # Phase 2: Key pressing config
│   ├── g1_piano_keys_env.py                # Phase 2: Key pressing env
│   └── agents/
│       ├── __init__.py
│       ├── rl_games_ppo_cfg.yaml           # RL Games training config
│       └── rsl_rl_ppo_cfg.py               # RSL-RL training config
├── assets/
│   ├── g1_piano_scene.py                   # Scene asset definitions
│   └── piano_with_keys.py                  # Phase 2: Detailed piano asset
├── scripts/
│   ├── train_piano_reach.py                # Training script
│   └── play_piano_reach.py                 # Inference/evaluation script
└── tutorials/00_sim/
    └── g1-piano-play.py                     # Original demo (keep as reference)
```

---

## Phase 1: Simple Piano Touch Task

### 1.1 Environment Configuration (`g1_piano_reach_env_cfg.py`)

```python
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

@configclass
class G1PianoReachEnvCfg(DirectRLEnvCfg):
    """Configuration for G1 Piano Reaching environment."""
    
    # Environment settings
    decimation = 2  # Control frequency = 60Hz (120Hz sim / 2)
    episode_length_s = 10.0  # 10 seconds per episode
    
    # Action space: 10 DOF (5 per arm)
    # left/right: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_roll
    action_space = 10
    action_scale = 0.5  # Scale down actions for smooth motion
    
    # Observation space (computed in env)
    # - Joint positions (10) - arm joints
    # - Joint velocities (10) - arm joints
    # - Hand positions relative to piano (6) - left/right hand xyz
    # - Previous action (10)
    observation_space = 36
    
    state_space = 0  # No privileged observations
    
    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,  # 120Hz physics
        render_interval=decimation,
        physics_material=...,  # Default friction
    )
    
    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,  # Start with 1024 parallel envs
        env_spacing=3.0,  # 3 meters between environments
        replicate_physics=True,
        clone_in_fabric=True,  # Fast cloning
    )
    
    # Robot configuration
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/G1_Robot",
        spawn=...,  # G1 spawn configuration
        actuators={
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*shoulder.*", ".*elbow.*"],
                stiffness=200.0,
                damping=20.0,
            ),
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"],
                stiffness=400.0,
                damping=40.0,
            ),
        },
    )
    
    # Piano target configuration
    piano_position = [0.0, -0.2, 0.5]  # Relative to G1's position
    piano_size = [0.5, 0.3, 0.1]  # Width, depth, height
    
    # Target points on piano (where hands should reach)
    # These are relative to piano center
    left_hand_target_offset = [-0.15, 0.0, 0.05]  # Left side of keyboard
    right_hand_target_offset = [0.15, 0.0, 0.05]  # Right side of keyboard
    
    # Reward scales
    rew_scale_reaching = 2.0  # Reward for reducing distance to piano
    rew_scale_contact = 10.0  # Bonus for touching piano
    rew_scale_both_hands = 5.0  # Extra bonus if both hands touch
    rew_scale_action_rate = -0.01  # Penalize large action changes (smooth motion)
    rew_scale_joint_limit = -0.5  # Penalize approaching joint limits
    rew_scale_posture = 0.5  # Reward for maintaining sitting posture
    
    # Termination conditions
    contact_force_threshold = 5.0  # Newton - what counts as "contact"
    max_hand_height = 1.5  # Reset if hands go too high
    min_torso_height = 0.3  # Reset if robot falls
    
    # Reset conditions
    reset_on_piano_contact = False  # Don't reset when touching piano (Phase 1)
    success_bonus = 20.0  # Bonus when both hands touch piano
```

### 1.2 Environment Implementation (`g1_piano_reach_env.py`)

Key structure based on CartPole example:

```python
from isaaclab.envs import DirectRLEnv
import torch

class G1PianoReachEnv(DirectRLEnv):
    cfg: G1PianoReachEnvCfg
    
    def __init__(self, cfg: G1PianoReachEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Cache joint indices for fast access
        self._arm_joint_indices = self._get_arm_joint_indices()
        self._leg_joint_indices = self._get_leg_joint_indices()
        
        # Cache body indices (for getting hand positions)
        self._left_hand_idx = self.robot.body_names.index("left_palm_link")
        self._right_hand_idx = self.robot.body_names.index("right_palm_link")
        
        # Target positions (world frame)
        self._compute_target_positions()
        
        # Sitting pose (fixed, not trainable)
        self._sitting_joint_positions = self._get_sitting_pose()
        
        # Track previous actions for smoothness reward
        self.previous_actions = torch.zeros((self.num_envs, 10), device=self.device)
        
    def _setup_scene(self):
        """Create the scene with G1 robot, piano, bench, table."""
        
        # Spawn G1 robot
        self.robot = Articulation(self.cfg.robot_cfg)
        
        # Spawn bench (procedural cuboid)
        self._spawn_bench()
        
        # Spawn table
        self._spawn_table()
        
        # Spawn piano (simple collision mesh for Phase 1)
        self._spawn_piano()
        
        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Filter collisions for CPU
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        
        # Add robot to scene
        self.scene.articulations["robot"] = self.robot
        
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step."""
        # Scale actions
        self.actions = self.cfg.action_scale * actions.clone()
    
    def _apply_action(self):
        """Apply actions to robot."""
        # Arm actions (trainable)
        self.robot.set_joint_position_target(
            self.actions, 
            joint_ids=self._arm_joint_indices
        )
        
        # Leg actions (fixed sitting pose)
        self.robot.set_joint_position_target(
            self._sitting_joint_positions,
            joint_ids=self._leg_joint_indices
        )
    
    def _get_observations(self) -> dict:
        """Compute observations for policy."""
        
        # Get joint states (arms only)
        arm_pos = self.robot.data.joint_pos[:, self._arm_joint_indices]
        arm_vel = self.robot.data.joint_vel[:, self._arm_joint_indices]
        
        # Get hand positions in world frame
        left_hand_pos = self.robot.data.body_pos_w[:, self._left_hand_idx]
        right_hand_pos = self.robot.data.body_pos_w[:, self._right_hand_idx]
        
        # Compute relative positions to targets
        left_hand_to_target = left_hand_pos - self.left_hand_target_w
        right_hand_to_target = right_hand_pos - self.right_hand_target_w
        
        # Concatenate observations
        obs = torch.cat([
            arm_pos,  # 10
            arm_vel,  # 10
            left_hand_to_target,  # 3
            right_hand_to_target,  # 3
            self.previous_actions,  # 10
        ], dim=-1)
        
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        return compute_rewards(
            # Hand positions
            self.robot.data.body_pos_w[:, self._left_hand_idx],
            self.robot.data.body_pos_w[:, self._right_hand_idx],
            # Target positions
            self.left_hand_target_w,
            self.right_hand_target_w,
            # Contact forces (detect touch)
            self._get_hand_contact_forces(),
            # Actions
            self.actions,
            self.previous_actions,
            # Joint positions (for limit penalty)
            self.robot.data.joint_pos[:, self._arm_joint_indices],
            # Reward scales
            self.cfg.rew_scale_reaching,
            self.cfg.rew_scale_contact,
            self.cfg.rew_scale_both_hands,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_joint_limit,
            self.cfg.contact_force_threshold,
        )
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        
        # Time-out
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Termination conditions
        torso_too_low = self.robot.data.root_pos_w[:, 2] < self.cfg.min_torso_height
        hands_too_high = torch.any(
            self.robot.data.body_pos_w[:, [self._left_hand_idx, self._right_hand_idx], 2] 
            > self.cfg.max_hand_height,
            dim=1
        )
        
        terminated = torso_too_low | hands_too_high
        
        return terminated, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specific environments."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        super()._reset_idx(env_ids)
        
        # Reset robot to sitting pose
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._leg_joint_indices] = self._sitting_joint_positions[env_ids]
        
        # Randomize arm positions slightly
        joint_pos[:, self._arm_joint_indices] += torch.randn_like(
            joint_pos[:, self._arm_joint_indices]
        ) * 0.1
        
        joint_vel = torch.zeros_like(joint_pos)
        
        # Reset root state
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        # Write to simulation
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Reset tracking variables
        self.previous_actions[env_ids] = 0


@torch.jit.script
def compute_rewards(
    left_hand_pos: torch.Tensor,
    right_hand_pos: torch.Tensor,
    left_target_pos: torch.Tensor,
    right_target_pos: torch.Tensor,
    contact_forces: torch.Tensor,  # (num_envs, 2) - left and right hand
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    joint_pos: torch.Tensor,
    rew_scale_reaching: float,
    rew_scale_contact: float,
    rew_scale_both_hands: float,
    rew_scale_action_rate: float,
    rew_scale_joint_limit: float,
    contact_threshold: float,
) -> torch.Tensor:
    """Compute reward components."""
    
    # Distance to targets (negative reward, want to minimize)
    left_dist = torch.norm(left_hand_pos - left_target_pos, dim=-1)
    right_dist = torch.norm(right_hand_pos - right_target_pos, dim=-1)
    
    # Distance reward (exponential to encourage getting close)
    dist_reward = -rew_scale_reaching * (left_dist + right_dist)
    
    # Contact detection
    left_contact = (contact_forces[:, 0] > contact_threshold).float()
    right_contact = (contact_forces[:, 1] > contact_threshold).float()
    
    # Contact reward
    contact_reward = rew_scale_contact * (left_contact + right_contact)
    
    # Both hands bonus
    both_hands_reward = rew_scale_both_hands * (left_contact * right_contact)
    
    # Action smoothness (penalize large changes)
    action_diff = torch.sum(torch.square(actions - previous_actions), dim=-1)
    action_penalty = rew_scale_action_rate * action_diff
    
    # Joint limit penalty (penalize approaching limits)
    # Assume normalized joint positions [-1, 1]
    near_limits = torch.sum(torch.square(torch.clamp(torch.abs(joint_pos) - 0.9, min=0.0)), dim=-1)
    joint_limit_penalty = rew_scale_joint_limit * near_limits
    
    # Total reward
    total_reward = (
        dist_reward + 
        contact_reward + 
        both_hands_reward + 
        action_penalty + 
        joint_limit_penalty
    )
    
    return total_reward
```

---

## Training Configuration

### RL Games PPO Config (`agents/rl_games_ppo_cfg.yaml`)

**References**: This config is based on Isaac Lab's Direct RL examples:
- **Primary**: `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/franka_cabinet/agents/rl_games_ppo_cfg.yaml` (manipulation task)
- **Secondary**: `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid/agents/rl_games_ppo_cfg.yaml` (humanoid complexity)
- **Baseline**: `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/agents/rl_games_ppo_cfg.yaml` (simple structure)

**Key Adaptations**:
- Network size `[256, 128, 64]` from Franka Cabinet (good for manipulation)
- `horizon_length: 16` from Franka (shorter episodes for reaching)
- `learning_rate: 3e-4` (slightly conservative)
- `max_epochs: 2000` (more than Franka's 1500, as piano reach may be harder)

```yaml
params:
  seed: 42
  
  algo:
    name: a2c_continuous
  
  model:
    name: continuous_a2c_logstd
  
  network:
    name: actor_critic
    separate: False
    
    space:
      continuous:
        mu_activation: None
        sigma_activation: None
        mu_init:
          name: default
        sigma_init:
          name: const_initializer
          val: 0.0
        fixed_sigma: True
    
    mlp:
      units: [256, 128, 64]
      activation: elu
      d2rl: False
      
      initializer:
        name: default
      regularizer:
        name: None
  
  load_checkpoint: False
  load_path: ''
  
  config:
    name: g1_piano_reach
    env_name: rlgpu
    multi_gpu: False
    ppo: True
    mixed_precision: False
    normalize_input: True
    normalize_value: True
    value_bootstrap: True
    num_actors: 1024
    reward_shaper:
      scale_value: 0.1
    normalize_advantage: True
    gamma: 0.99
    tau: 0.95
    learning_rate: 3e-4
    lr_schedule: adaptive
    schedule_type: standard
    kl_threshold: 0.016
    score_to_win: 20000
    max_epochs: 2000
    save_best_after: 100
    save_frequency: 100
    print_stats: True
    grad_norm: 1.0
    entropy_coef: 0.0
    truncate_grads: True
    e_clip: 0.2
    horizon_length: 16
    minibatch_size: 8192
    mini_epochs: 8
    critic_coef: 4
    clip_value: True
    seq_len: 4
    bounds_loss_coef: 0.0001
```

---

## Training Script (`scripts/train_piano_reach.py`)

```python
#!/usr/bin/env python3

import argparse
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Train G1 to reach piano")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments")
parser.add_argument("--task", type=str, default="Isaac-Piano-Reach-G1-v0", help="Task name")
parser.add_argument("--headless", action="store_true", help="Run headless")
parser.add_argument("--video", action="store_true", help="Record video")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after app launch
import gymnasium as gym
from rl_games.torch_runner import Runner

def main():
    # Create environment
    env = gym.make(args.task, num_envs=args.num_envs, device=args.device)
    
    # Create RL Games runner
    runner = Runner()
    runner.load_config({
        # Load from yaml
        "params": {...}  # Load from agents/rl_games_ppo_cfg.yaml
    })
    runner.reset()
    runner.run({
        "train": True,
        "play": False,
        "checkpoint": "",
        "sigma": None
    })
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
```

---

## Expected Results & Timeline

### Phase 1: Simple Piano Touch

**Week 1-2: Setup & Initial Training**
- Implement environment configuration and class
- Debug observation/reward shapes
- Train for 500-1000 epochs
- Expected: Hands move toward piano, may not touch yet

**Week 2-3: Reward Tuning**
- Adjust reward scales
- Add curriculum (start with closer piano, move farther)
- Expected: Consistent touches, but maybe one hand only

**Week 3-4: Polish & Validation**
- Tune for both-hand contact
- Add visualization
- Record videos
- **Goal: 80%+ success rate on both-hand touch**

**Training Metrics to Track:**
- Average distance to piano (should decrease)
- Contact rate (should increase to ~80%+)
- Episode length (should increase as robot learns not to fall)
- Both-hands success rate (final metric)

---

## Phase 2: Piano Key Pressing

### 2.1 Enhanced Environment

**Key Changes:**

1. **Replace Piano Asset**
   - Current: Single collision mesh
   - New: 88 individual keys, each with collision sensor
   - Each key has: position, size, contact sensor

2. **Modified Observation Space**
   - Add: Target key ID (one-hot encoding or embedding)
   - Add: All key states (pressed or not) - optional
   - Keep: Joint positions, velocities, hand-to-target distance

3. **Modified Reward Structure**
   ```python
   # Phase 2 rewards
   rew_reach_correct_key = 2.0  # Reaching toward correct key
   rew_press_correct_key = 10.0  # Pressing correct key
   rew_wrong_key_penalty = -5.0  # Touching wrong key
   rew_key_press_depth = 1.0  # Reward for pressing down (not just touching)
   ```

4. **Curriculum Learning**
   - Stage 1: Only press middle C (single key)
   - Stage 2: Press any of 10 middle keys
   - Stage 3: Press any key on command
   - Stage 4: Press sequences of keys

### 2.2 Implementation Strategy

**Option A: Separate Env Class (Recommended)**
```python
class G1PianoKeysEnv(G1PianoReachEnv):
    """Extends Phase 1 env with key pressing."""
    
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._setup_keys()
    
    def _get_observations(self):
        obs = super()._get_observations()
        # Add target key info
        obs["policy"] = torch.cat([
            obs["policy"],
            self.target_key_embedding  # Which key to press
        ], dim=-1)
        return obs
    
    def _get_rewards(self):
        # New reward function with key-specific logic
        return compute_key_rewards(...)
```

**Option B: Unified Env with Config Flag**
```python
class G1PianoEnv(DirectRLEnv):
    def __init__(self, cfg, **kwargs):
        if cfg.task_mode == "reach":
            # Phase 1 logic
        elif cfg.task_mode == "keys":
            # Phase 2 logic
```

---

## Key Technical Challenges & Solutions

### Challenge 1: Getting Hand Positions

**Solution**:
```python
# In __init__
self._left_hand_idx = self.robot.find_bodies("left_palm_link")[0]
self._right_hand_idx = self.robot.find_bodies("right_palm_link")[0]

# In _get_observations or _get_rewards
left_hand_pos_w = self.robot.data.body_pos_w[:, self._left_hand_idx]
right_hand_pos_w = self.robot.data.body_pos_w[:, self._right_hand_idx]
```

**Alternative using body_link_pose_w**:
```python
# Get full pose (position + orientation)
left_hand_pose = self.robot.data.body_link_pose_w[:, self._left_hand_idx]  # Shape: (num_envs, 7)
left_hand_pos = left_hand_pose[:, :3]  # Position
left_hand_quat = left_hand_pose[:, 3:7]  # Orientation (w,x,y,z)
```

### Challenge 2: Detecting Piano Contact

**Solution 1: Contact Sensor (Preferred)**
```python
from isaaclab.sensors import ContactSensor, ContactSensorCfg

# In _setup_scene
self.contact_sensor_cfg = ContactSensorCfg(
    prim_path="/World/envs/env_.*/G1_Robot/.*palm.*",
    update_period=0.0,  # Update every step
    history_length=1,
    track_air_time=False,
)
self.contact_sensor = ContactSensor(self.contact_sensor_cfg)
self.scene.sensors["hand_contact"] = self.contact_sensor

# In _get_rewards or helper function
def _get_hand_contact_forces(self):
    contact_data = self.contact_sensor.data
    # Extract force magnitude for left and right hands
    left_force = contact_data.net_forces_w[:, 0].norm(dim=-1)
    right_force = contact_data.net_forces_w[:, 1].norm(dim=-1)
    return torch.stack([left_force, right_force], dim=-1)
```

**Solution 2: Distance-Based (Simpler, Less Accurate)**
```python
# Define success as distance < threshold
success_dist = 0.02  # 2cm
left_success = (torch.norm(left_hand_pos - left_target_pos, dim=-1) < success_dist)
right_success = (torch.norm(right_hand_pos - right_target_pos, dim=-1) < success_dist)
```

### Challenge 3: Maintaining Sitting Pose

**Solution: Dual Action Space**
```python
# Legs are controlled by fixed PD targets (from sitting pose)
# Arms are controlled by RL policy

def _apply_action(self):
    # Apply RL actions to arms
    self.robot.set_joint_position_target(
        self.actions,  # From policy
        joint_ids=self._arm_joint_indices
    )
    
    # Apply fixed sitting pose to legs
    self.robot.set_joint_position_target(
        self._sitting_joint_positions.expand(self.num_envs, -1),
        joint_ids=self._leg_joint_indices
    )
```

**Alternative: Include Legs in Observation, Add Posture Reward**
```python
# In _get_rewards
posture_deviation = torch.sum(
    (leg_joint_pos - sitting_leg_pos) ** 2,
    dim=-1
)
posture_penalty = -0.1 * posture_deviation  # Small penalty
```

### Challenge 4: Sparse Rewards (Hands Don't Reach Piano)

**Solution: Reward Shaping with Dense Distance Rewards**
```python
# Exponential distance reward (gets large when close)
dist_reward = torch.exp(-3.0 * distance_to_target)

# Or negative distance (linear)
dist_reward = -1.0 * distance_to_target

# Or inverse distance
dist_reward = 1.0 / (1.0 + distance_to_target)
```

**Solution: Curriculum Learning**
```python
# Start with piano very close, gradually move farther
class CurriculumManager:
    def __init__(self):
        self.piano_distance = 0.3  # Start close
    
    def update(self, success_rate):
        if success_rate > 0.8:
            self.piano_distance += 0.05  # Move farther
            self.piano_distance = min(self.piano_distance, 0.6)  # Max distance
```

---

## Comparison with Animation Approach

| Aspect | RL Approach | Animation/IK Approach |
|--------|-------------|----------------------|
| **Development Time** | 3-6 weeks | 1-2 weeks |
| **Flexibility** | High - can adapt to different pianos | Low - fixed to scene |
| **Realism** | High - emergent behavior | Medium - scripted |
| **Robustness** | High - learns to recover | Low - brittle to changes |
| **Transferability** | Can transfer to real robot | Difficult to transfer |
| **Phase 2 (Keys)** | Natural extension | Requires significant rework |
| **Computational Cost** | High during training | Low (no training) |
| **Final Policy Size** | Small (~10MB network) | N/A (procedural) |

---

## Next Steps (In Order)

### Immediate (This Week)

1. **Create environment structure**
   ```bash
   mkdir -p g1-piano-play/envs/agents
   mkdir -p g1-piano-play/scripts
   ```

2. **Copy and adapt cartpole example**
   - Start with `cartpole_env.py` as template
   - Replace cartpole logic with G1 + piano scene
   - Test scene creation first (no RL yet)

3. **Verify hand position extraction**
   ```python
   # Test script to print hand positions
   left_hand_idx = robot.find_bodies("left_palm_link")[0]
   print(f"Left hand: {robot.data.body_pos_w[:, left_hand_idx]}")
   ```

4. **Create minimal environment**
   - Just reach task, simple distance reward
   - No contact sensing yet
   - 16 envs for fast testing

### Week 1

5. **Test observation space**
   - Print observation shapes
   - Verify ranges are reasonable
   - Test reset functionality

6. **Test reward function**
   - Hard-code hand movements
   - Verify rewards increase when closer
   - Plot reward components

7. **First training run**
   - Train for 100 epochs
   - Check for NaN/Inf
   - Verify learning curve exists

### Week 2-3

8. **Tune hyperparameters**
   - Adjust reward scales
   - Try different network sizes
   - Experiment with learning rates

9. **Add contact sensing**
   - Implement ContactSensor
   - Add contact rewards
   - Test touch detection

10. **Scale up training**
    - Increase to 1024+ envs
    - Train for 1000+ epochs
    - Monitor success rate

### Week 4

11. **Evaluation & visualization**
    - Create play/inference script
    - Record videos
    - Measure success metrics

12. **Documentation**
    - Document final hyperparameters
    - Create usage guide
    - Prepare for Phase 2

---

## Resources & References

### Isaac Lab Direct RL Environment Examples

**Manipulation Tasks** (Most relevant):
- **Franka Cabinet**: `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/franka_cabinet/`
  - `franka_cabinet_env.py` - Reaching and opening drawer
  - `agents/rl_games_ppo_cfg.yaml` - Training config (PRIMARY REFERENCE)
  - Good for: Environment structure, reaching rewards, contact detection
  
- **In-Hand Manipulation**: `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/inhand_manipulation/`
  - `inhand_manipulation_env.py` - Object manipulation with hands
  - Good for: Contact-based rewards, distance rewards, success criteria

**Humanoid Tasks**:
- **Humanoid**: `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid/`
  - `humanoid_env.py` - Humanoid locomotion
  - Good for: Humanoid robot control, network sizing, reward scaling

**Basic Examples**:
- **Cartpole**: `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/`
  - `cartpole_env.py` - Simplest DirectRLEnv example
  - Good for: Understanding basic structure, quick reference

### Isaac Lab Core APIs

**Environment Framework**:
- `IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py` - Base DirectRLEnv class
- `IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env_cfg.py` - Base config class

**Robot Control**:
- `IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py` - Articulation class
- `IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation_data.py` - State access (body positions, joint states)

**Sensors**:
- `IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor.py` - Contact force detection
- `IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor_cfg.py` - Contact sensor config

**Scene Management**:
- `IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene.py` - Multi-environment handling
- `IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene_cfg.py` - Scene configuration

### Unitree G1 Resources
- **G1 Configurations**: `IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py`
  - Available models: `G1_CFG`, `G1_MINIMAL_CFG`, `G1_29DOF_CFG`
  - Joint names, actuator configs, spawn configurations
  
- **Unitree Sim (External)**: `unitree_sim_isaaclab/` (separate project in your workspace)
  - `sim_main.py` - Reference implementation
  - Pre-trained policies in `assets/model/`

### RL Training Frameworks

**RL Games** (Recommended for Isaac Lab):
- GitHub: https://github.com/Denys88/rl_games
- Documentation: https://rl-games.readthedocs.io/
- Config examples: https://github.com/Denys88/rl_games/tree/master/rl_games/configs
- Isaac Lab integration: `IsaacLab/source/extensions/isaaclab.rl_games/`

**RSL-RL** (Lightweight alternative):
- GitHub: https://github.com/leggedrobotics/rsl_rl
- Isaac Lab integration: `IsaacLab/source/extensions/isaaclab.rsl_rl/`
- Used by: ETH Zurich's legged robotics lab

**Stable-Baselines3** (Popular, well-documented):
- Documentation: https://stable-baselines3.readthedocs.io/
- Isaac Lab integration: `IsaacLab/source/extensions/isaaclab.sb3/`

### Learning Resources

**Isaac Lab Tutorials**:
- Creating Direct RL Env: `IsaacLab/docs/source/tutorials/03_envs/create_direct_rl_env.rst`
- Running RL Training: `IsaacLab/docs/source/tutorials/03_envs/run_rl_training.rst`
- Environment Design: `IsaacLab/docs/source/setup/walkthrough/api_env_design.rst`

**RL Theory** (if needed):
- OpenAI Spinning Up: https://spinningup.openai.com/en/latest/
- PPO Paper: https://arxiv.org/abs/1707.06347
- Isaac Gym Paper: https://arxiv.org/abs/2108.10470 (foundation of Isaac Lab)

**Isaac Lab Main Documentation**:
- Main docs: https://isaac-sim.github.io/IsaacLab/
- API reference: https://isaac-sim.github.io/IsaacLab/source/api/index.html
- GitHub: https://github.com/isaac-sim/IsaacLab

### Command-Line Tools

**Training with RL Games**:
```bash
# Basic training
./isaaclab.sh -p scripts/train.py --task Isaac-Piano-Reach-G1-v0

# With custom config
./isaaclab.sh -p scripts/train.py --task Isaac-Piano-Reach-G1-v0 \
  --num_envs 1024 --headless

# Resume from checkpoint
./isaaclab.sh -p scripts/train.py --task Isaac-Piano-Reach-G1-v0 \
  --checkpoint runs/experiment_name/nn/model.pth --resume
```

**Viewing Results**:
```bash
# TensorBoard
tensorboard --logdir runs/

# Play trained policy
./isaaclab.sh -p scripts/play.py --task Isaac-Piano-Reach-G1-v0 \
  --checkpoint /path/to/model.pth --num_envs 1
```

### Helper Scripts & Tools

**Environment Testing**:
```bash
# Test environment creation
./isaaclab.sh -p scripts/test_env.py

# Profile performance
./isaaclab.sh -p scripts/profile_env.py --num_envs 1024
```

**Debug Tools**:
- Print all joint names: Check robot.data.joint_names
- Print all body names: Check robot.body_names
- Visualize forces: Enable contact force visualization in Isaac Sim viewer

### Related Projects (in your workspace)

**Unitree Sim Isaac Lab**:
- Location: `/home/solotech007/RoboGym/simulation/unitree_sim_isaaclab/`
- Main file: `sim_main.py`
- Can reference for: G1 model usage, DDS integration, locomotion policies

---

## Success Criteria

### Phase 1 Complete When:
- ✅ Both hands touch piano targets 80%+ of episodes
- ✅ Average episode length > 8 seconds (out of 10s max)
- ✅ Training is stable (no reward collapse)
- ✅ Policy works across all 1024 parallel environments
- ✅ Inference script can load and run policy
- ✅ Video recordings show smooth, natural motion

### Phase 2 Complete When:
- ✅ Can press specific keys on command 70%+ accuracy
- ✅ Wrong key penalty prevents random pressing
- ✅ Can handle at least 50% of all piano keys
- ✅ Key press sequences work (multi-step episodes)

---

## Conclusion

**Recommendation**: Proceed with RL approach

**Rationale**:
1. More interesting research problem
2. Better foundation for Phase 2 (key pressing)
3. Transferable to real robot in future
4. Demonstrates RL capability (good for portfolio/publication)
5. Aligns with Isaac Lab's strengths

**Estimated Total Time**:
- Phase 1: 3-4 weeks
- Phase 2: Additional 3-4 weeks
- **Total: 6-8 weeks** for fully working piano key pressing

**Risk Level**: Medium
- Higher than animation approach
- But with clear milestones and fallback options
- Large body of Isaac Lab examples to reference

Ready to start implementation! 🚀🤖🎹

