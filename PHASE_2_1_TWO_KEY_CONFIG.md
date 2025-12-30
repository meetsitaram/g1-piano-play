# Phase 2.1: Two-Key Configuration

## 🎯 Purpose

Start with **2 large, well-separated keys** (C and D) for initial sequential learning. This provides:
- ✅ Easier target acquisition (larger keys)
- ✅ Clearer distinction between targets (more separation)
- ✅ Faster learning and debugging
- ✅ Proof of concept for sequential task

Once mastered, progressively reduce key size and spacing toward realistic piano dimensions.

---

## 🎹 Key Specifications

### Phase 2.1: Initial Learning (2 Keys)

**Key Dimensions**:
- **Width**: 250mm (0.25m) - **2.5x larger** than final
- **Length**: 200mm (0.2m) - same as Phase 1 piano depth
- **Height**: 100mm (0.1m) - same as Phase 1 piano

**Key Spacing**:
- **Gap between keys**: 200mm (0.2m) - **4x larger** than final

**Total Piano Width**:
- 2 × 0.25m + 1 × 0.2m = **0.7m** (well within G1's reach)

**Key Names**: ["C", "D"]

**Visual Layout** (top view):
```
        G1 Robot (sitting on bench)
              👤
              |
              |
         [   C   ]     [   D   ]
         =========     =========
          250mm         250mm
         <------- 200mm gap ------->
         <------- 0.7m total ------->
```

**Position on Table**:
- **C key center**: x = -0.225m (left of center)
- **D key center**: x = +0.225m (right of center)
- **Y position**: -0.21m (same as Phase 1, toward robot)
- **Z position**: On table top (0.48m)

---

## 📊 Progressive Scaling Plan

### Stage 1: Large & Separated (Phase 2.1)
```
Keys: C, D
Width: 250mm (2.5x target)
Spacing: 200mm (4x target)
Total width: 0.7m
Episode: 10 seconds (5s per key)
Success criteria: 90% completion rate
```

### Stage 2: Medium (Phase 2.2) 
```
Keys: C, D, E
Width: 150mm (1.5x target)
Spacing: 100mm (2x target)
Total width: 0.65m
Episode: 15 seconds
Success criteria: 85% completion rate
```

### Stage 3: Near-Final (Phase 2.3)
```
Keys: C, D, E, F, G
Width: 120mm (1.2x target)
Spacing: 70mm (1.4x target)
Total width: 0.88m
Episode: 25 seconds
Success criteria: 80% completion rate
```

### Stage 4: Final Dimensions (Phase 2.4)
```
Keys: C, D, E, F, G, A, B
Width: 100mm (1.0x - realistic)
Spacing: 50mm (1.0x - realistic)
Total width: 1.05m
Episode: 40 seconds
Success criteria: 70% completion rate
```

---

## 🎨 Visual Debugging

### Key Colors (Phase 2.1)
- **C key**: Red (1.0, 0.0, 0.0)
- **D key**: Orange (1.0, 0.5, 0.0)

Large color contrast for easy visual distinction during training.

### Target Highlighting (Optional)
- Current target key glows/pulses brighter
- Helps visualize which key agent should press

---

## 📐 Coordinate System

### World Frame (from G1's perspective)
```
        +Z (up)
         |
         |
    +X --|-- -X (left to right)
         |
        -Y (forward toward piano)
```

### Key Positions (Phase 2.1)

**C Key** (left):
- Center: `(-0.225, -0.21, 0.48)` in env-local frame
- Edges: x from -0.35m to -0.1m

**D Key** (right):
- Center: `(+0.225, -0.21, 0.48)` in env-local frame
- Edges: x from +0.1m to +0.35m

**Separation**: 200mm gap between keys (0.1m to -0.1m in X)

---

## 🧪 Reachability Analysis

### G1 Sitting Pose
- Torso at: `(0.0, -0.8, 0.6)` on bench
- Arms at rest: hands near hips

### Arm Reach (Estimated)
- **Forward reach** (Y-axis): ~0.6-0.8m
- **Lateral reach** (X-axis): ~±0.4m from body center
- **Vertical reach** (Z-axis): 0.4-0.8m

### Key Positions Relative to G1
- **Piano Y**: -0.21m (table edge toward robot)
- **G1 Y**: -0.8m (on bench)
- **Distance**: 0.59m forward ✅ (within reach)

- **C key X**: -0.225m
- **D key X**: +0.225m
- **Lateral spread**: 0.45m ✅ (within reach)

**Verdict**: Both keys easily reachable! ✅

---

## 🎓 Training Configuration (Phase 2.1)

### Episode Structure
```python
episode_length_s = 10.0  # Total episode time
time_per_key = 5.0       # 5 seconds per key
num_keys = 2             # C, D

Timeline:
  0.0 -  5.0s: Target = C (press key C)
  5.0 - 10.0s: Target = D (press key D)
```

### Observation Space (37 dims)
```python
obs = [
    arm_joint_pos (10),          # Robot state
    arm_joint_vel (10),
    target_key_index (1),        # 0=C, 1=D (normalized: 0.0 or 1.0)
    left_hand_to_target (3),     # Distance to current target key
    right_hand_to_target (3),
    left_hand_to_C (1),          # Distance to C key
    left_hand_to_D (1),          # Distance to D key
    right_hand_to_C (1),         # Distance to C key
    right_hand_to_D (1),         # Distance to D key
    previous_actions (10),
]
```

### Reward Function
```python
reward = (
    # PRIMARY: Reach correct key
    + distance_to_correct_key * 5.0       # Approach target
    + correct_key_contact * 25.0          # Touch correct key (HIGH)
    + sequence_completion * 75.0          # Pressed key, moving to next
    
    # PENALTIES: Wrong behavior
    - wrong_key_contact * 15.0            # Touched wrong key
    - idle_penalty * 0.1                  # Not progressing
    
    # SMOOTHNESS: Same as Phase 1
    - action_rate * 0.5
    - joint_vel * 0.05
    - joint_accel * 0.01
    
    # BONUS: Complete sequence
    + full_sequence_bonus * 100.0         # Pressed both C and D in order
)
```

### Success Metrics
- **Keys pressed per episode**: Average should approach 2.0
- **Correct sequence rate**: 90%+ episodes press C then D
- **Wrong key rate**: <5% of key presses
- **Average reward**: +800-1000 per episode

---

## 🔧 Implementation Files

### New Files to Create
1. `envs/g1_piano_sequence_env.py` - Sequential key environment
2. `envs/g1_piano_sequence_env_cfg.py` - Configuration
3. `envs/agents/rl_games_ppo_sequence_cfg.yaml` - PPO config for Phase 2
4. `scripts/test_key_reachability.py` - Visualization/test script

### Modified Files
1. `envs/__init__.py` - Register new environment

---

## 🚀 Quick Start Commands

### 1. Test Key Reachability (Visual)
```bash
cd /home/solotech007/RoboGym/simulation
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/test_key_reachability.py \
    --num_keys 2 \
    --key_width 0.25 \
    --key_spacing 0.2
```

Watch G1 and 2 large keys spawn. Verify keys are reachable.

### 2. Train Phase 2.1 (2 Keys)
```bash
# Visual training (16 envs)
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py \
    --task Isaac-Piano-Sequence-G1-2Key-v0 \
    --num_envs 16

# Full training (2048 envs, headless)
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py \
    --task Isaac-Piano-Sequence-G1-2Key-v0 \
    --num_envs 2048 \
    --headless
```

### 3. Test Trained Policy
```bash
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/test_policy_visual.py \
    --checkpoint runs/YOUR_RUN/nn/last_*.pth \
    --num_envs 4 \
    --episode_length 120 \
    --num_episodes 5
```

Watch robot press C, then D, repeatedly!

---

## 📈 Expected Training Progress

### Phase 2.1 Training Curve
```
Reward
  |     
+1000|              ___________  ← Success (both keys)
     |           .-'
+500 |        .-'
     |     .-'
   0 |  .-'
     |.'
-500 |'___________________________
      0   200  400  600  800  1000
               Epochs
```

**Milestones**:
- **0-100 epochs**: Learning to press first key (C)
- **100-300 epochs**: Learning to press second key (D)
- **300-500 epochs**: Learning sequence (C then D)
- **500+ epochs**: Consistent success (90%+ rate)

---

## ✅ Success Criteria (Before Moving to Phase 2.2)

Phase 2.1 is complete when:
- ✅ **90%+ episodes** press both C and D in correct order
- ✅ **Wrong key rate** < 5%
- ✅ **Average episode reward** > +800
- ✅ **Visual inspection** shows smooth, deliberate reaching
- ✅ **Sequence timing** correct (presses C at 0-5s, D at 5-10s)

---

## 🎯 Next Phase (Phase 2.2)

After mastering 2 large keys, we'll:
1. Add 3rd key (E)
2. Reduce key width: 250mm → 150mm
3. Reduce spacing: 200mm → 100mm
4. Increase episode: 10s → 15s

This progressive scaling ensures robust learning at each stage.

---

## 📝 Notes

### Why Start Large?
- **Easier target acquisition**: Large keys are harder to miss
- **Faster learning**: More reward signal early on
- **Better debugging**: Easy to see which key was pressed
- **Confidence building**: Proves sequential approach works

### Why Reduce Gradually?
- **Curriculum learning**: Progressive difficulty scaling
- **Avoid catastrophic forgetting**: Small changes preserve learned behavior
- **Fine motor control**: Gradually learn precision
- **Smooth transition**: Each stage builds on previous

---

Ready to implement? Let's start with the reachability test! 🎹✨

