# Phase 2: Sequential Piano Key Pressing - Implementation Plan

## 🎯 Milestone Goal

Train G1 humanoid robot to press **specific piano keys in a predefined sequence** (C → D → E → F → G → A → B) over a 40-second episode, with 5 seconds allocated per key.

---

## 📋 Current State (Phase 1)

✅ **Achieved**:
- G1 can reach and touch a single piano cuboid with both hands
- Smooth, controlled arm movements
- High success rate (+4000 reward)
- Stable sitting position

**Current Setup**:
- Single piano cuboid (0.5m × 0.2m × 0.1m)
- Binary reward: "touching piano" vs "not touching"
- No key differentiation
- 10-second episodes

---

## 🎹 Phase 2 Requirements

### 1. **Piano Representation** (Progressive Key Sizes)

**UPDATED: Starting with 2 large keys, progressively scaling to realistic dimensions**

**Phase 2.1 (Initial - 2 Keys)**:
- **Number of keys**: 2 (C, D)
- **Key dimensions**: 
  - Width: **250mm (0.25m)** per key - **2.5x larger** than final
  - Length: 200mm (0.2m) - same as current piano depth
  - Height: 100mm (0.1m) - same as current piano
- **Spacing**: **200mm (0.2m)** gap between keys - **4x larger** than final
- **Total piano width**: 2 × 0.25m + 1 × 0.2m = **0.7m** (easily reachable)
- **Episode**: 10 seconds (5s per key)

**Phase 2.4 (Final - 7 Keys)**:
- **Number of keys**: 7 (C, D, E, F, G, A, B - one octave, white keys only)
- **Key dimensions**: 
  - Width: 100mm (0.1m) per key
  - Length: 200mm (0.2m) - same as current piano depth
  - Height: 100mm (0.1m) - same as current piano
- **Spacing**: 50mm (0.05m) gap between adjacent keys
- **Total piano width**: 7 × 0.1m + 6 × 0.05m = 1.0m
- **Key names**: ["C", "D", "E", "F", "G", "A", "B"]
- **Episode**: 40 seconds (5s per key)

**Positioning** (left to right from G1's perspective):
```
  C    D    E    F    G    A    B
[■]  [■]  [■]  [■]  [■]  [■]  [■]
```

Each key is a separate `MeshCuboidCfg` with unique prim path and color for visual debugging.

---

### 2. **Episode Structure** (40 seconds, 7 keys)

**Timing**:
- Total episode: 40 seconds (4800 steps at 120Hz)
- Time per key: 5 seconds (~600 steps)
- Sequence: C (0-5s) → D (5-10s) → E (10-15s) → F (15-20s) → G (20-25s) → A (25-30s) → B (30-35s)
- Buffer: Last 5 seconds for final key press completion

**Sequence Progression**:
- Start of episode: Target = C
- After 5s: Target = D
- After 10s: Target = E
- ... and so on

**Success Criteria**:
- Episode is successful if G1 touches **all 7 keys in correct order**
- Touching a wrong key → penalty (but episode continues)
- Missing a key entirely → lower reward (but episode continues)

---

### 3. **Observation Space Expansion**

**Current (Phase 1)**: 36 dimensions
```python
obs = [
    arm_joint_pos (10),        # Current arm positions
    arm_joint_vel (10),        # Current arm velocities
    left_hand_to_target (3),   # Distance to single target
    right_hand_to_target (3),  # Distance to single target
    previous_actions (10),     # Action history
]
```

**New (Phase 2)**: **46 dimensions** (+10)
```python
obs = [
    # Robot state (20 dims)
    arm_joint_pos (10),              # Current arm positions
    arm_joint_vel (10),              # Current arm velocities
    
    # Target information (18 dims)
    target_key_one_hot (7),          # Which key to press (C=1000000, D=0100000, etc.)
    left_hand_to_target_key (3),    # Distance to current target key
    right_hand_to_target_key (3),   # Distance to current target key
    time_in_sequence (1),            # Normalized time (0.0 to 1.0) within current key window
    sequence_progress (1),           # Which key in sequence (0-6, normalized to 0-1)
    time_remaining_in_episode (1),  # Normalized episode time remaining
    keys_pressed_so_far (1),         # Count of successfully pressed keys
    
    # Action history (10 dims)
    previous_actions (10),           # Action history for smoothness
]
```

**Alternative (Simpler, 43 dims)**:
```python
obs = [
    arm_joint_pos (10),
    arm_joint_vel (10),
    target_key_index (1),            # 0-6 for C-B (instead of one-hot)
    left_hand_to_target_key (3),
    right_hand_to_target_key (3),
    hand_to_all_keys (14),           # Distance from left/right hand to all 7 keys (7×2)
    previous_actions (10),
]
```

**Recommendation**: Start with **simpler 43-dim** version, can expand later if needed.

---

### 4. **Reward Function Redesign**

**Phase 1 Rewards** (Current):
```python
reward = (
    distance_to_piano * 5.0           # Get closer to piano
    + contact_bonus * 15.0             # Touch piano
    + both_hands_bonus * 10.0          # Both hands touch
    + smoothness_penalties             # Smooth motion
)
```

**Phase 2 Rewards** (Sequential Key Pressing):
```python
reward = (
    # PRIMARY: Correct key interaction
    + distance_to_correct_key * 5.0       # Approach current target key
    + correct_key_contact * 20.0          # Touch correct key (HIGH reward)
    + sequence_completion_bonus * 50.0    # Bonus when progressing to next key
    
    # PENALTIES: Wrong behavior
    - wrong_key_contact * 10.0            # Touched wrong key (PENALTY)
    - idle_penalty * 0.1                  # Not moving toward target
    
    # SECONDARY: Motion quality
    + smoothness_penalties                # Same as Phase 1
    + joint_limit_penalty                 # Same as Phase 1
    
    # BONUS: Full sequence completion
    + full_sequence_bonus * 200.0         # Completed all 7 keys in order (sparse)
)
```

**Key Reward Components**:

1. **Distance to Correct Key** (`+5.0`):
   - Only reward getting closer to the **current target key**
   - Ignore distance to other keys
   
2. **Correct Key Contact** (`+20.0`):
   - Large bonus for touching the correct key
   - Only awarded when touching the **current target**
   
3. **Sequence Completion Bonus** (`+50.0`):
   - One-time bonus when agent successfully presses current key and moves to next
   - Helps reinforce "key press → move to next key" behavior
   
4. **Wrong Key Penalty** (`-10.0`):
   - Penalize touching any key that is NOT the current target
   - Teaches agent to avoid wrong keys
   
5. **Idle Penalty** (`-0.1`):
   - Small penalty if not moving toward target (distance not decreasing)
   - Encourages continuous progress
   
6. **Full Sequence Bonus** (`+200.0`):
   - Massive bonus for completing entire sequence (all 7 keys)
   - Sparse reward, but helps shape final policy

---

### 5. **Key Contact Detection**

**Phase 1**: Single piano contact (distance-based)
```python
contact = (distance_to_piano < 0.05)  # 5cm threshold
```

**Phase 2**: Multi-key contact detection

**Option A: Distance-Based (Simple, Recommended for MVP)**
```python
# For each hand, find closest key
left_hand_distances_to_keys = [
    distance(left_hand, key_C),
    distance(left_hand, key_D),
    # ... for all 7 keys
]
closest_key_left = argmin(left_hand_distances_to_keys)
left_contact = (left_hand_distances_to_keys[closest_key_left] < 0.05)

# Check if contacted key matches target
if left_contact and closest_key_left == current_target_key_index:
    reward += correct_key_contact_bonus
elif left_contact:
    reward += wrong_key_penalty
```

**Option B: ContactSensor (Advanced, Phase 3)**
- Add `ContactSensor` to each key prim
- Detect actual physical contact forces
- More accurate but complex to set up
- Recommended for later refinement

**Recommendation**: Use **distance-based (Option A)** initially, upgrade to ContactSensor in Phase 3.

---

### 6. **Sequence Management**

**Tracking Current Target**:
```python
class G1PianoSequenceEnv(DirectRLEnv):
    def __init__(self, ...):
        # Define key sequence
        self.key_sequence = [0, 1, 2, 3, 4, 5, 6]  # C, D, E, F, G, A, B
        self.key_names = ["C", "D", "E", "F", "G", "A", "B"]
        
        # Time per key (in seconds)
        self.time_per_key = 5.0
        
        # Track progress per environment
        self.current_key_index = torch.zeros(num_envs, dtype=torch.long)  # 0-6
        self.keys_pressed_count = torch.zeros(num_envs, dtype=torch.long)  # 0-7
        self.key_press_success = torch.zeros((num_envs, 7), dtype=torch.bool)  # Track which keys pressed
        
    def _compute_rewards(self):
        # Get current target key for each env
        current_step = self.episode_length_buf  # Steps since reset
        current_time = current_step / self.physics_dt  # Seconds
        
        # Calculate which key should be active now
        target_key_index = torch.floor(current_time / self.time_per_key).long()
        target_key_index = torch.clamp(target_key_index, 0, 6)  # Max index = 6 (key B)
        
        # Check if hands are touching current target key
        # ... (contact detection logic)
        
        # Award bonus if key pressed successfully
        if contact_with_target and not self.key_press_success[env_id, target_key_index]:
            reward += sequence_completion_bonus
            self.key_press_success[env_id, target_key_index] = True
            self.keys_pressed_count[env_id] += 1
```

**Time-Based Progression**:
- Episode step count determines current target key
- Every 5 seconds (600 steps), target key changes automatically
- Agent doesn't need to explicitly "advance" to next key

**Alternative (Event-Based Progression)**:
- Key changes only when agent successfully presses current key
- More flexible but harder to learn (agent controls pacing)
- Recommended for **Phase 3** after mastering time-based

---

## 🏗️ Implementation Steps

### Step 1: **Environment Structure** (New File)
Create `envs/g1_piano_sequence_env.py` and `envs/g1_piano_sequence_env_cfg.py`:
- Inherit from Phase 1 environment
- Override key spawning, observation, and reward functions
- Keep sitting pose, actuators, and smoothness logic

### Step 2: **Piano Key Spawning**
```python
def _spawn_piano_keys(self):
    """Spawn 8 individual piano keys (C, D, E, F, G, A, B)."""
    key_names = ["C", "D", "E", "F", "G", "A", "B"]
    key_width = 0.1  # 100mm
    key_spacing = 0.05  # 50mm gap
    total_width = 7 * key_width + 6 * key_spacing  # 1.0m
    
    # Center the piano on table
    start_x = -total_width / 2.0  # Left edge
    
    self.key_positions = []
    for i, key_name in enumerate(key_names):
        key_x = start_x + i * (key_width + key_spacing) + key_width / 2.0
        key_y = -0.21  # Same as current piano
        key_z = table_height + 0.05  # On table
        
        # Spawn key cuboid
        key_cfg = sim_utils.MeshCuboidCfg(
            size=(key_width, 0.2, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=self._get_key_color(i)  # Different color per key
            ),
            # ... collision and physics props
        )
        key_cfg.func(f"/World/envs/env_.*/PianoKey_{key_name}", key_cfg, translation=(key_x, key_y, key_z))
        
        # Store key center position for distance calculations
        self.key_positions.append((key_x, key_y, key_z))
    
    self.key_positions = torch.tensor(self.key_positions, device=self.device)  # [7, 3]
```

### Step 3: **Observation Space**
```python
def _get_observations(self) -> dict:
    # Get robot state
    arm_pos = self.robot.data.joint_pos[:, self._arm_joint_indices]  # [N, 10]
    arm_vel = self.robot.data.joint_vel[:, self._arm_joint_indices]  # [N, 10]
    
    # Calculate current target key index based on time
    current_time = self.episode_length_buf * self.physics_dt
    target_key_index = torch.floor(current_time / self.cfg.time_per_key).long()
    target_key_index = torch.clamp(target_key_index, 0, 6)  # [N]
    
    # Get target key positions for each env
    target_key_pos = self.key_positions[target_key_index]  # [N, 3]
    target_key_pos_w = target_key_pos + self.scene.env_origins  # World frame
    
    # Hand positions
    left_hand_pos = self.robot.data.body_pos_w[:, self._left_hand_idx]  # [N, 3]
    right_hand_pos = self.robot.data.body_pos_w[:, self._right_hand_idx]  # [N, 3]
    
    # Distance to current target key
    left_to_target = target_key_pos_w - left_hand_pos  # [N, 3]
    right_to_target = target_key_pos_w - right_hand_pos  # [N, 3]
    
    # Distances to ALL keys (for richer observation)
    all_keys_pos_w = self.key_positions.unsqueeze(0) + self.scene.env_origins.unsqueeze(1)  # [N, 7, 3]
    left_to_all_keys = torch.norm(all_keys_pos_w - left_hand_pos.unsqueeze(1), dim=-1)  # [N, 7]
    right_to_all_keys = torch.norm(all_keys_pos_w - right_hand_pos.unsqueeze(1), dim=-1)  # [N, 7]
    
    # Concatenate observations
    obs = torch.cat([
        arm_pos,                              # 10
        arm_vel,                              # 10
        target_key_index.unsqueeze(-1).float() / 6.0,  # 1 (normalized)
        left_to_target,                       # 3
        right_to_target,                      # 3
        left_to_all_keys,                     # 7
        right_to_all_keys,                    # 7
        self.previous_actions,                # 10
    ], dim=-1)  # Total: 51 dims
    
    return {"obs": obs}
```

### Step 4: **Reward Function**
```python
def _get_rewards(self) -> torch.Tensor:
    # Determine current target key
    current_time = self.episode_length_buf * self.physics_dt
    target_key_index = torch.floor(current_time / self.cfg.time_per_key).long()
    target_key_index = torch.clamp(target_key_index, 0, 6)
    
    # Get target key positions in world frame
    target_key_pos_w = self.key_positions[target_key_index] + self.scene.env_origins
    
    # Hand positions
    left_hand_pos = self.robot.data.body_pos_w[:, self._left_hand_idx]
    right_hand_pos = self.robot.data.body_pos_w[:, self._right_hand_idx]
    
    # Distance to target key
    left_dist_to_target = torch.norm(left_hand_pos - target_key_pos_w, dim=-1)
    right_dist_to_target = torch.norm(right_hand_pos - target_key_pos_w, dim=-1)
    min_dist_to_target = torch.min(left_dist_to_target, right_dist_to_target)
    
    # Distance reward (encourage approaching target key)
    distance_reward = -self.cfg.rew_scale_distance * min_dist_to_target
    
    # Contact detection: which key is each hand touching?
    all_keys_pos_w = self.key_positions.unsqueeze(0) + self.scene.env_origins.unsqueeze(1)  # [N, 7, 3]
    left_dists_to_all = torch.norm(all_keys_pos_w - left_hand_pos.unsqueeze(1), dim=-1)  # [N, 7]
    right_dists_to_all = torch.norm(all_keys_pos_w - right_hand_pos.unsqueeze(1), dim=-1)  # [N, 7]
    
    left_contact_key = torch.argmin(left_dists_to_all, dim=-1)  # [N] - which key is closest
    right_contact_key = torch.argmin(right_dists_to_all, dim=-1)  # [N]
    
    left_is_contact = left_dists_to_all[torch.arange(self.num_envs), left_contact_key] < 0.05  # [N]
    right_is_contact = right_dists_to_all[torch.arange(self.num_envs), right_contact_key] < 0.05  # [N]
    
    # Correct key contact reward
    left_correct = (left_contact_key == target_key_index) & left_is_contact
    right_correct = (right_contact_key == target_key_index) & right_is_contact
    correct_contact_reward = self.cfg.rew_scale_correct_contact * (left_correct.float() + right_correct.float())
    
    # Wrong key penalty
    left_wrong = (left_contact_key != target_key_index) & left_is_contact
    right_wrong = (right_contact_key != target_key_index) & right_is_contact
    wrong_contact_penalty = self.cfg.rew_scale_wrong_contact * (left_wrong.float() + right_wrong.float())
    
    # Sequence completion bonus (one-time per key)
    sequence_bonus = torch.zeros(self.num_envs, device=self.device)
    for env_id in range(self.num_envs):
        target_idx = target_key_index[env_id].item()
        if (left_correct[env_id] or right_correct[env_id]) and not self.key_press_success[env_id, target_idx]:
            sequence_bonus[env_id] = self.cfg.rew_scale_sequence_completion
            self.key_press_success[env_id, target_idx] = True
            self.keys_pressed_count[env_id] += 1
    
    # Full sequence completion bonus (sparse)
    full_sequence_bonus = (self.keys_pressed_count == 7).float() * self.cfg.rew_scale_full_sequence
    
    # Smoothness penalties (same as Phase 1)
    action_penalty = self.cfg.rew_scale_action_rate * torch.sum(torch.square(self.actions - self.previous_actions), dim=-1)
    # ... (velocity and acceleration penalties)
    
    # Total reward
    total_reward = (
        distance_reward +
        correct_contact_reward +
        wrong_contact_penalty +
        sequence_bonus +
        full_sequence_bonus +
        action_penalty
        # + other penalties
    )
    
    return total_reward
```

### Step 5: **Configuration Updates**
```python
# In g1_piano_sequence_env_cfg.py

class G1PianoSequenceEnvCfg(G1PianoReachEnvCfg):  # Inherit from Phase 1
    # Observation/Action
    observation_space = 51  # Updated
    action_space = 10  # Same
    
    # Episode timing
    episode_length_s = 40.0  # 40 seconds for full sequence
    time_per_key = 5.0  # 5 seconds per key
    
    # Piano keys
    num_keys = 7
    key_names = ["C", "D", "E", "F", "G", "A", "B"]
    key_width = 0.1  # 100mm
    key_spacing = 0.05  # 50mm gap
    
    # Rewards (REBALANCED for sequential task)
    rew_scale_distance = 5.0              # Approach target key
    rew_scale_correct_contact = 20.0      # Touch correct key (HIGH)
    rew_scale_wrong_contact = -10.0       # Touch wrong key (PENALTY)
    rew_scale_sequence_completion = 50.0  # Complete one key in sequence
    rew_scale_full_sequence = 200.0       # Complete all 7 keys (sparse)
    
    # Smoothness (same as Phase 1)
    rew_scale_action_rate = -0.5
    rew_scale_joint_vel = -0.05
    rew_scale_joint_accel = -0.01
    
    # Contact detection
    contact_distance_threshold = 0.05  # 5cm
```

### Step 6: **Training Script Update**
```python
# Register new environment in envs/__init__.py
gym.register(
    id="Isaac-Piano-Sequence-G1-v0",  # NEW
    entry_point="envs.g1_piano_sequence_env:G1PianoSequenceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1PianoSequenceEnvCfg,
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_sequence_cfg.yaml",
    },
)
```

### Step 7: **Visual Debugging**
Add different colors to each key for easy identification:
```python
def _get_key_color(self, key_index):
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
```

Additionally, highlight the **current target key** by making it brighter or pulsing (optional, for debugging).

---

## 🎓 Training Strategy

### Curriculum Learning (Recommended)

Training directly on all 7 keys may be too hard initially. Use **progressive curriculum**:

#### **Stage 1: 2 Keys (C, D)** - 10 seconds episode
- Simplest: Just learn to press C, then D
- Episode: 5s per key
- Success: Can press both keys in order consistently

#### **Stage 2: 3 Keys (C, D, E)** - 15 seconds episode
- Add one more key
- Success: 90%+ success rate on 3-key sequence

#### **Stage 3: 5 Keys (C, D, E, F, G)** - 25 seconds episode
- Majority of keyboard
- Success: 80%+ success rate on 5-key sequence

#### **Stage 4: 7 Keys (Full Sequence)** - 40 seconds episode
- Complete sequence
- Success: 70%+ success rate on full 7-key sequence

**Implementation**:
- Create separate env configs for each stage
- Use `--checkpoint` to load previous stage's policy
- Fine-tune on harder task

**Alternative** (More Complex):
- Implement **dynamic curriculum** that automatically increases difficulty based on performance
- Track success rate and add more keys when agent reaches threshold

### Transfer Learning from Phase 1

**Option 1**: Start from scratch with new observation space (51 dims vs 36 dims)
- Clean slate, but slower initial learning

**Option 2**: Initialize arm control weights from Phase 1 policy
- Extract policy network weights from Phase 1 checkpoint
- Initialize first layers of Phase 2 network
- Fine-tune on sequential task
- **Recommended for faster convergence**

---

## 🔧 Hyperparameter Adjustments

Phase 2 is more complex, may need tuning:

### Episode Length
```yaml
# Longer episodes for sequential task
max_episode_length: 4800  # 40 seconds at 120Hz (vs 1200 for Phase 1)
```

### Horizon Length
```yaml
# May need longer horizons for temporal credit assignment
horizon_length: 32  # vs 16 in Phase 1
# This means batch_size = num_envs × 32
```

### Learning Rate
```yaml
# Start lower for stability with sequential task
learning_rate: 5e-5  # vs 1e-4 in Phase 1
# Can increase later if learning is too slow
```

### Reward Scales
- May need to tune reward balance after initial training
- Monitor which keys are being pressed (or missed)
- Adjust `rew_scale_wrong_contact` if agent touches wrong keys too often

---

## 📊 Success Metrics

### Training Metrics (TensorBoard)
1. **`rewards/correct_key_contact`**: Should increase over time
2. **`rewards/wrong_key_penalty`**: Should decrease (fewer wrong touches)
3. **`rewards/sequence_completion`**: Increasing = more keys pressed
4. **`rewards/full_sequence`**: Sparse, but should appear eventually
5. **`info/keys_pressed_per_episode`**: Track average keys pressed (0-7)
6. **`info/sequence_success_rate`**: % of episodes completing full sequence

### Visual Testing
- Watch robot press keys in sequence
- Verify correct keys are touched at correct times
- Check motion is still smooth (no flapping)

### Quantitative Goals
- **Early training (0-500 epochs)**: Press 2-3 keys consistently
- **Mid training (500-1500 epochs)**: Press 4-5 keys consistently
- **Late training (1500+ epochs)**: Press all 7 keys in 80%+ of episodes

---

## 🚧 Potential Challenges & Solutions

### Challenge 1: **Sparse Rewards**
- **Problem**: Agent may struggle to press first key, never gets reward signal
- **Solution**: Add **dense shaping rewards**:
  - Reward for reducing distance to target key (done)
  - Reward for hand velocity toward target (optional)
  - Curriculum learning (start with 2 keys)

### Challenge 2: **Temporal Credit Assignment**
- **Problem**: Agent presses key at wrong time (e.g., presses E when should press C)
- **Solution**: 
  - Include time information in observations (done)
  - Use longer horizon_length for better temporal understanding
  - May need RNN/LSTM policy (advanced)

### Challenge 3: **Key Spacing & Reachability**
- **Problem**: Keys may be too far apart for G1 to reach all of them
- **Solution**:
  - Verify total piano width (1.15m) fits within G1's arm reach
  - If not, reduce key_spacing or num_keys
  - Test with manual reaching simulation first

### Challenge 4: **Both Hands vs Single Hand**
- **Problem**: Should both hands press same key, or alternate?
- **Solution** (Current approach): Allow either hand to press any key
  - More flexible, agent figures out optimal strategy
  - Alternative: Assign specific hands to specific keys (more constraint)

### Challenge 5: **Training Time**
- **Problem**: Sequential task is much harder, may take 5-10x longer to train
- **Solution**:
  - Use curriculum learning (2 keys → 7 keys)
  - Transfer learning from Phase 1
  - Consider reducing num_keys to 5 initially

### Challenge 6: **Episode Termination**
- **Problem**: Should episode end if agent touches wrong key?
- **Solution** (Current approach): No, episode continues but with penalty
  - Allows agent to recover and complete sequence
  - Alternative: Early termination for wrong key (harsher, faster learning but less robust)

---

## 🎨 Visual Enhancements (Optional)

### 1. **Target Key Highlighting**
- Make current target key glow or pulse
- Helps with visual debugging and demo videos

### 2. **Progress Indicator**
- Show which keys have been successfully pressed (e.g., turn white after press)
- Visual feedback for successful sequence

### 3. **Hand Trails**
- Draw lines showing hand paths during episode
- Useful for analyzing motion patterns

---

## 📝 Implementation Checklist

### Phase 2A: MVP (Minimum Viable Product)
- [ ] Create `g1_piano_sequence_env.py` and `g1_piano_sequence_env_cfg.py`
- [ ] Implement 7-key piano spawning with correct spacing
- [ ] Update observation space (51 dims)
- [ ] Implement sequence management (time-based progression)
- [ ] Implement multi-key contact detection (distance-based)
- [ ] Implement sequential reward function
- [ ] Add key press tracking (`key_press_success`)
- [ ] Create PPO config for Phase 2
- [ ] Register new environment
- [ ] Test environment with random actions (verify keys spawn correctly)
- [ ] Train on 2-key sequence (C, D) - 500 epochs
- [ ] Verify agent can press both keys in order

### Phase 2B: Full Sequence
- [ ] Increase to 3 keys (C, D, E) - train 500 epochs
- [ ] Increase to 5 keys (C, D, E, F, G) - train 1000 epochs
- [ ] Train on full 7-key sequence - train 2000 epochs
- [ ] Achieve 70%+ full sequence completion rate
- [ ] Visual testing and demo recording

### Phase 2C: Refinement (Optional)
- [ ] Add ContactSensor for more accurate key detection
- [ ] Implement dynamic curriculum learning
- [ ] Add visual key highlighting for target key
- [ ] Optimize reward weights based on training results
- [ ] Transfer learning from Phase 1 weights

---

## 🔄 Suggested Improvements to Your Approach

### 1. **Start with Fewer Keys**
**Your plan**: 7 keys immediately (C, D, E, F, G, A, B)

**Suggestion**: Start with **2-3 keys**, scale up progressively
- Reason: 7 keys in 40s is very challenging for initial learning
- Agent may fail to press even first key → no reward signal
- Easier to debug with fewer keys

**Revised Timeline**:
- Week 1: 2 keys (C, D)
- Week 2: 3 keys (C, D, E)
- Week 3: 5 keys
- Week 4: 7 keys (full sequence)

---

### 2. **Consider Hand Specialization**
**Your plan**: Implicit - both hands reach any key

**Suggestion**: Add **hand assignment** to observation/rewards
- Left hand presses C, D, E (left side)
- Right hand presses F, G, A, B (right side)
- Speeds up learning by reducing action space complexity
- More realistic for piano playing

**Implementation**:
```python
# In observation
assigned_hand_for_target_key = 0 or 1  # 0=left, 1=right

# In reward
if target_key in [0,1,2]:  # C, D, E
    reward_correct_hand = left_hand_contact * 20.0
    reward_wrong_hand = right_hand_contact * 5.0  # Still okay, but lower reward
```

---

### 3. **Key Width & Spacing**
**Your plan**: 100mm width, 50mm spacing

**Suggestion**: Verify this fits within G1's reach
- Total piano width: 1.15m
- G1 sitting arm reach: ~0.8-1.0m (estimate)
- **Concern**: Outer keys (C and B) may be unreachable

**Test First**:
1. Spawn 7 keys with your dimensions
2. Run `test_env.py` with fixed poses (arms fully extended left/right)
3. Check if hands can reach keys C and B
4. If not, adjust:
   - Reduce spacing: 50mm → 30mm
   - Reduce key width: 100mm → 80mm
   - Reduce num_keys: 7 → 5

---

### 4. **Episode Structure: Time-Based vs Event-Based**
**Your plan**: Time-based (5s per key, fixed)

**Suggestion**: Consider **hybrid approach**
- Time-based for initial training (simpler, your current plan)
- Event-based for final policy (more flexible)

**Event-Based** (Phase 3):
- Key changes only when agent successfully presses it
- Agent can press keys faster if skilled (e.g., 2s per key)
- More realistic for actual piano playing

---

### 5. **Add "Hold" Reward**
**Your plan**: Binary contact (touch = reward)

**Suggestion**: Reward **sustained contact**
- Piano keys need to be pressed for ~0.5s to register
- Add reward for holding key contact:
  ```python
  rew_scale_key_hold = 5.0  # Reward per step while holding correct key
  ```
- Encourages deliberate presses vs brief taps

---

### 6. **Wrong Key Strategy**
**Your plan**: Implicit - penalty for wrong key

**Suggestion**: Consider **early termination** option
- If agent presses wrong key 3+ times → episode ends early
- Teaches agent to be careful
- Speeds up training (avoids wasting steps on failed episodes)

**Trade-off**: Less robust policy (more brittle)

**Recommendation**: Start without early termination, add later if needed

---

## 🎯 Revised Implementation Plan (Recommended)

### Milestone 2.1: **Two-Key Sequence** (Week 1)
- Keys: C, D only
- Episode: 10 seconds (5s per key)
- Observation: 37 dims (simpler than full 51)
- Goal: 90% success rate pressing both keys in order

### Milestone 2.2: **Three-Key Sequence** (Week 2)
- Keys: C, D, E
- Episode: 15 seconds
- Observation: 41 dims
- Goal: 80% success rate

### Milestone 2.3: **Five-Key Sequence** (Week 3)
- Keys: C, D, E, F, G
- Episode: 25 seconds
- Observation: 47 dims
- Goal: 70% success rate

### Milestone 2.4: **Full Seven-Key Sequence** (Week 4)
- Keys: C, D, E, F, G, A, B
- Episode: 40 seconds (your original plan)
- Observation: 51 dims
- Goal: 60-70% success rate

---

## 📚 Documentation to Create

1. **`PHASE_2_SEQUENTIAL_KEYS_PLAN.md`** (this file)
2. **`PHASE_2_TRAINING_LOG.md`** - Track training progress per curriculum stage
3. **`KEY_REACHABILITY_TEST.md`** - Document which keys G1 can reach
4. **Updated `README.md`** - Add Phase 2 instructions

---

## 🚀 Next Steps (For Review)

1. **Review this plan** - any questions or concerns?
2. **Test key reachability** - verify 1.15m piano fits within G1's reach
3. **Decide on curriculum** - start with 2 keys or jump to 7?
4. **Implement Phase 2.1** - create basic 2-key environment
5. **Train and iterate** - scale up gradually

---

## ❓ Questions for You

1. **Curriculum**: Start with 2 keys and scale up, or jump to 7 immediately?
2. **Hand assignment**: Should left/right hands have assigned keys, or free choice?
3. **Key dimensions**: Are 100mm width + 50mm spacing realistic for G1's reach? (Need to verify)
4. **Episode termination**: Continue on wrong key press, or terminate early?
5. **Black keys**: Phase 3 should add black keys (sharps/flats), or stick with white keys only?
6. **Transfer learning**: Load Phase 1 weights to initialize Phase 2, or train from scratch?

Let me know your thoughts and we can refine the plan before implementation! 🎹

