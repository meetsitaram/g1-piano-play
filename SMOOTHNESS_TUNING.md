# Motion Smoothness Tuning Guide

## 🎯 Problem: Arm Flapping/Flickering

The robot was successfully reaching the piano but with **jerky, flapping motions** instead of smooth, natural movements. This is a common RL problem where the agent achieves the goal but doesn't care about movement efficiency.

---

## ⚠️ Critical Distinction: `action_scale` vs `action_rate`

**These are DIFFERENT and both can prevent reaching, but for different reasons:**

### 1. **`action_scale`** (Physical Limit - Hardware)
```python
action_scale = 0.5  # BAD: Physically limits joint commands
```
- **Effect**: Scales down action values BEFORE applying to joints
- **Problem**: Arms physically cannot move far enough (hardware constraint)
- **Example**: If policy outputs 1.0 rad, only 0.5 rad is applied → **can't reach piano**
- **Solution**: Set to `1.0` for full range

### 2. **`rew_scale_action_rate`** (Economic Limit - Reward)
```python
rew_scale_action_rate = -5.0  # BAD: Too strong penalty
```
- **Effect**: Penalizes large action changes in reward function
- **Problem**: Agent learns to barely move to avoid penalty (economic constraint)
- **Example**: Policy learns "don't move much" is safer than "reach aggressively"
- **Solution**: Balance with strong reaching rewards

### ✅ **Correct Configuration:**
```python
action_scale = 1.0                 # Full physical range
rew_scale_reaching = 5.0           # Strong reaching motivation
rew_scale_action_rate = -0.5       # Moderate smoothness penalty
```

**Key:** Reaching reward (`5.0`) must dominate smoothness penalty (`-0.5`) so agent prioritizes reaching despite penalties!

---

## ✅ Solutions Implemented

### 1. **Increased Action Rate Penalty** (Most Important)

**Change:**
```python
rew_scale_action_rate = -1.0  # Increased from -0.01 (100x increase!)
```

**Effect:**
- Penalizes large changes in actions between consecutive timesteps
- Forces the policy to make gradual, smooth adjustments
- **This is the primary fix for flapping behavior**

**Tuning Guide:**
- **Too low** (`-0.01` to `-0.1`): Arms flap wildly
- **Good range** (`-0.5` to `-2.0`): Smooth but responsive
- **Too high** (`-5.0+`): Arms move too slowly, won't reach piano

---

### 2. **Joint Velocity Penalty** (New)

**Change:**
```python
rew_scale_joint_vel = -0.05  # Penalize high arm velocities
```

**Effect:**
- Penalizes rapid joint movements directly
- Encourages slower, more controlled motion
- Complements action rate penalty

**Tuning Guide:**
- **Recommended** (`-0.01` to `-0.1`): Gentle smoothing
- **Too high** (`-0.5+`): Arms barely move

---

### 3. **Joint Acceleration Penalty** (New, Optional)

**Change:**
```python
rew_scale_joint_accel = -0.01  # Penalize velocity changes
```

**Effect:**
- Penalizes rapid changes in velocity (jerk)
- Makes motion even smoother by avoiding sudden accelerations
- **Can be disabled** by setting to `0.0` if too constraining

**Tuning Guide:**
- **Subtle effect** (`-0.001` to `-0.05`): Minor smoothing
- **Can disable** by setting to `0.0` if motion is too sluggish

---

## 🔧 How to Tune for Your Needs

### Scenario 1: **Still Too Flappy After Training**

Increase action rate penalty:
```python
rew_scale_action_rate = -2.0  # or even -3.0
```

### Scenario 2: **Arms Move Too Slowly Now**

Reduce penalties:
```python
rew_scale_action_rate = -0.5
rew_scale_joint_vel = -0.02
rew_scale_joint_accel = 0.0  # Disable
```

### Scenario 3: **Perfect Speed but Still Jerky**

Keep action rate high, add more velocity penalty:
```python
rew_scale_action_rate = -1.0  # Keep
rew_scale_joint_vel = -0.1    # Increase
```

### Scenario 4: **Robot-Like Motion (Too Smooth)**

Reduce all smoothness penalties:
```python
rew_scale_action_rate = -0.3
rew_scale_joint_vel = -0.01
rew_scale_joint_accel = 0.0
```

---

## 📊 Training Impact

### Expected Changes:
- **Training time**: May increase by 10-20% (smoother policies take longer to learn)
- **Final reward**: May be slightly lower initially (more penalties), but should stabilize
- **Convergence**: Should be more stable (less oscillation in training curves)

### What to Monitor:
```bash
tensorboard --logdir runs/
```

Watch these metrics:
- **`rewards/total`**: Should still increase toward positive values
- **`rewards/action_penalty`**: Will be more negative (expected)
- **`rewards/joint_vel_penalty`**: New metric to monitor
- **Episode length**: Should stay similar

---

## 🚀 Training Commands

### Test with Few Envs First (Visualize):
```bash
cd /home/solotech007/RoboGym/simulation
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --num_envs 16 \
    --headless
```

### Full Training (Headless):
```bash
cd /home/solotech007/RoboGym/simulation
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --num_envs 2048 \
    --headless
```

---

## 🎨 Visual Testing

After training a few hundred epochs, test the smoothness:

```bash
cd /home/solotech007/RoboGym/simulation
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/test_policy_visual.py \
    --checkpoint runs/YOUR_RUN/nn/last_*.pth \
    --num_envs 4 \
    --episode_length 150 \
    --num_episodes 3
```

**Look for:**
- ✅ Smooth, flowing arm movements
- ✅ Gradual speed changes
- ✅ Natural reaching motion
- ❌ No rapid oscillations
- ❌ No jerky starts/stops

---

## 📝 Technical Details

### Reward Function Components (Updated):

```python
total_reward = (
    dist_reward              # Encourage reaching piano
    + contact_reward         # Bonus for touching
    + both_hands_reward      # Extra bonus for both hands
    + action_penalty         # NEW: Smoothness (action rate)
    + joint_vel_penalty      # NEW: Smoothness (velocity)
    + joint_accel_penalty    # NEW: Smoothness (acceleration)
    + joint_limit_penalty    # Safety (avoid limits)
)
```

### Smoothness Penalties Explained:

1. **Action Penalty**: `Σ(action_t - action_{t-1})²`
   - Measures how much commands change between steps
   - Most direct way to reduce flapping

2. **Velocity Penalty**: `Σ(joint_vel)²`
   - Measures actual joint speeds
   - Encourages slower movements

3. **Acceleration Penalty**: `Σ(joint_vel_t - joint_vel_{t-1})²`
   - Measures velocity changes (jerk)
   - Makes motion even smoother

---

## ⚠️ CRITICAL: Reward Balance

**Important:** Smoothness penalties must NOT prevent the robot from reaching the piano!

### The Balance Equation:
```
Reaching Rewards >> Smoothness Penalties
```

If penalties are too strong, the agent learns to "play it safe" by barely moving, never reaching the piano.

### Current Balanced Settings:
```python
# POSITIVE REWARDS (encourage reaching)
rew_scale_reaching = 5.0      # Distance improvement (DOMINANT)
rew_scale_contact = 15.0      # Contact bonus (STRONG)
rew_scale_both_hands = 10.0   # Both hands bonus

# NEGATIVE PENALTIES (discourage flapping)
rew_scale_action_rate = -0.5  # Smoothness (MODERATE)
rew_scale_joint_vel = -0.05   # Velocity penalty (SMALL)
rew_scale_joint_accel = -0.01 # Acceleration penalty (TINY)
```

**Key insight:** Distance reward (`5.0`) is **10x stronger** than action penalty (`-0.5`), ensuring the agent prioritizes reaching over standing still.

---

## 🎯 Recommended Starting Values (UPDATED AFTER TESTING)

**⚠️ CRITICAL LESSON LEARNED:** Velocity/acceleration penalties accumulate MASSIVELY over episodes and prevent reaching!

For **smooth piano reaching** (allows reaching while reducing flapping):

```python
# In g1_piano_reach_env_cfg.py

# POSITIVE REWARDS (must be STRONG)
rew_scale_reaching = 10.0         # Strong motivation to reach (was 2.0 → 5.0 → 10.0)
rew_scale_contact = 20.0          # Very strong contact reward (was 10.0 → 15.0 → 20.0)
rew_scale_both_hands = 15.0       # Strong coordination bonus (was 5.0 → 10.0 → 15.0)

# SMOOTHNESS PENALTIES (must be GENTLE)
rew_scale_action_rate = -0.1      # Gentle penalty, only stops extreme flapping
rew_scale_joint_vel = 0.0         # DISABLED (was -0.05, too strong!)
rew_scale_joint_accel = 0.0       # DISABLED (was -0.01, too strong!)
```

### Why Velocity/Acceleration Penalties Failed:

**Math**: With 10 arm joints moving at 2 rad/s:
```
vel_penalty = -0.05 × (10 joints × 2² rad/s²) = -2.0 per step
Over 1200 steps (10s episode) = -2400 total penalty!
```

This **overwhelms** the reaching rewards, causing robot to learn "don't move."

**Solution**: Use only action rate penalty (measures command changes, not actual movement).

For **faster task completion** (efficient but less smooth):

```python
rew_scale_action_rate = -0.3      # Less constraint
rew_scale_joint_vel = -0.01       # Minimal penalty
rew_scale_joint_accel = 0.0       # Disabled
```

---

## 🔄 Iterative Tuning Process

1. **Train** for 200-500 epochs with new settings
2. **Visualize** checkpoint with `test_policy_visual.py`
3. **Assess** motion smoothness visually
4. **Adjust** penalties based on observation
5. **Repeat** until satisfied

### Quick Iteration Tip:
Train with **16 envs + visualization** for faster feedback on motion quality before committing to full 2048-env training.

---

## 🚨 Troubleshooting: "Arms Not Reaching Piano"

### Symptom:
After adding smoothness penalties, arms barely move or don't reach the piano.

### Diagnosis:
Check if **smoothness penalties are too strong** relative to reaching rewards.

### Quick Test:
Monitor TensorBoard → `rewards/total`:
- **Negative or zero**: Penalties dominating → increase reaching rewards
- **Positive and increasing**: Good balance → keep training
- **Flat/stuck**: Agent gave up → rebalance rewards

### Fix Options:

#### Option 1: **Increase Reaching Rewards** (Recommended)
```python
rew_scale_reaching = 10.0   # Even stronger motivation
rew_scale_contact = 20.0    # Much larger bonus
```

#### Option 2: **Reduce Smoothness Penalties**
```python
rew_scale_action_rate = -0.3  # Gentler penalty
rew_scale_joint_vel = -0.02   # Minimal penalty
rew_scale_joint_accel = 0.0   # Disable
```

#### Option 3: **Verify Physical Range**
```python
action_scale = 1.0  # Must be 1.0 for full range
```

### Success Criteria:
After retraining, you should see:
- ✅ **Positive total reward** (reaching dominates penalties)
- ✅ **Arms extend fully** toward piano
- ✅ **Both hands touch** piano frequently
- ✅ **Smooth motion** (not jerky, but still reaches)

---

## 📚 References

- **Action smoothness**: Common in manipulation tasks (Isaac Lab FrankaCabinet, Reach)
- **Joint velocity penalties**: Used in locomotion tasks (Unitree Go1/H1)
- **Acceleration penalties**: Advanced smoothing (less common but effective)

---

## ✅ Expected Outcome

After retraining with these settings, you should see:
- 🎯 **Still reaches piano** (positive rewards)
- ✨ **Reduced flapping** (gentler than before, but not perfectly smooth)
- ✅ **Successful contact** (primary goal maintained)

**Trade-off**: With only action rate penalty, motion won't be perfectly smooth, but robot WILL reach the piano. This is the correct priority!

---

## 🚨 CRITICAL: Troubleshooting Negative Rewards

### Symptom: Training stuck at large negative rewards (e.g., -3700)

**What's happening**:
```
rewards/total: -3700 (should be positive!)
rewards/iter: flat, not improving
Robot barely moving in visual tests
```

**Diagnosis**: Smoothness penalties are TOO STRONG, robot learned "don't move."

**Root Cause**: Velocity/acceleration penalties accumulate over entire episode:
```python
# BAD: This penalty accumulates every step!
vel_penalty = -0.05 × sum(joint_velocities²)  # -2.0 per step
Over 1200 steps = -2400 total!  # Overwhelms reaching rewards
```

**Quick Fix**:
1. **Disable velocity/acceleration penalties**:
   ```python
   rew_scale_joint_vel = 0.0      # Was -0.05
   rew_scale_joint_accel = 0.0    # Was -0.01
   ```

2. **Increase reaching rewards**:
   ```python
   rew_scale_reaching = 10.0      # Was 5.0
   rew_scale_contact = 20.0       # Was 15.0
   ```

3. **Reduce action penalty** (if still stuck):
   ```python
   rew_scale_action_rate = -0.1   # Was -0.5
   ```

4. **Retrain from scratch** (old checkpoint learned bad policy)

**Verification**: After 100-200 epochs, reward should be **positive** (+500 to +2000)

---

## 🎓 Key Lesson Learned

**Smoothness penalties must be MUCH gentler than reaching rewards!**

| Approach | Result |
|----------|--------|
| **Phase 1** (no smoothness) | ✅ +4750 reward, ❌ flappy motion |
| **Attempt 1** (vel/accel penalties) | ❌ -3700 reward, robot won't move |
| **Attempt 2** (only action rate, gentle) | ✅ +3000-4000 reward, ✅ reaches piano, ⚠️ some flapping |

**Conclusion**: Use **ONLY action rate penalty** at **-0.1 to -0.3** for manipulation tasks. Velocity/acceleration penalties are too harsh for episodic tasks.

