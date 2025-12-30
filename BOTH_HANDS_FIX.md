# Fix: One Hand Only Problem

## 🎯 Problem

After fixing the velocity penalties, training showed:
- ✅ **Positive rewards** (+3000-4000)
- ✅ **Smoother hand motion**
- ❌ **One hand settles close to piano, other hand stays far away**

**Root Cause**: **Local Optimum** - robot discovered "one hand is good enough" strategy.

---

## 🔍 Why This Happened

### Previous Reward Structure (Suboptimal):

```python
# One hand touching:
contact_reward = 20.0 × 1 hand = +20
total = +20

# Both hands touching:
contact_reward = 20.0 × 2 hands = +40
both_hands_bonus = 15.0
total = +55

# Incremental benefit: +55 - 20 = +35 for second hand
```

**Problem**: Agent got +20 reward for one hand and thought "this is good enough!" The extra +35 for the second hand wasn't compelling enough to justify the effort and risk.

**RL Perspective**: 
- One hand = guaranteed +20 (stable local optimum)
- Two hands = potential +55 but requires more exploration and coordination
- Optimizer chose safe strategy: "settle with one hand"

---

## ✅ The Solution

### Three-Pronged Approach:

#### 1. **Reduce Single-Hand Appeal**
```python
rew_scale_contact = 10.0  # Was 20.0 → reduced by 50%
```
- One hand now only gives +10 (less appealing)
- Makes single-hand strategy less attractive

#### 2. **Massively Increase Both-Hands Bonus**
```python
rew_scale_both_hands = 50.0  # Was 15.0 → increased 3.3x!
```
- Both hands now give: 10×2 + 50 = **+70 total**
- Incremental benefit: +70 - 10 = **+60 for second hand** (huge!)

#### 3. **Add Asymmetry Penalty** (NEW)
```python
rew_scale_hand_asymmetry = -2.0  # Penalize distance difference
```
- Penalizes when one hand is close but other is far
- Penalty = -2.0 × |left_dist - right_dist|
- Example: If left hand is 0.05m away and right is 0.5m away:
  - Asymmetry penalty = -2.0 × |0.05 - 0.5| = -2.0 × 0.45 = **-0.9**
- Continuously encourages reducing hand distance difference

---

## 📊 Reward Comparison

| Scenario | Old Reward | New Reward | Change |
|----------|-----------|------------|--------|
| **No contact** | 0 | 0 | Same |
| **One hand only** | +20 | +10 | -50% (less appealing ✅) |
| **One hand + asymmetry** | +20 | +10 - 0.9 = +9.1 | Even worse! ✅ |
| **Both hands** | +55 | +70 | +27% (much better! ✅) |

**Key**: Incremental benefit for second hand:
- **Old**: +35 (from +20 to +55)
- **New**: +60 (from +10 to +70)
- **Improvement**: 71% more reward for getting both hands!

---

## 🧠 How This Changes Behavior

### Before (Local Optimum):
```
Agent thinking:
  "I have one hand touching (+20 reward)"
  "Moving second hand is risky and only gives +35 more"
  "Better to stay stable with +20"
```

**Result**: Settles with one hand, never explores both hands.

### After (Global Optimum):
```
Agent thinking:
  "I have one hand touching (+10 reward)"
  "But I'm getting asymmetry penalty (-0.9)"
  "Net: only +9.1"
  
  "If I get both hands: +70 reward!"
  "That's +60 more than current state!"
  "Worth the effort!"
```

**Result**: Strong motivation to coordinate both hands.

---

## 📐 Math Behind Asymmetry Penalty

### Formula:
```python
hand_distance_diff = |left_dist_to_target - right_dist_to_target|
asymmetry_penalty = -2.0 × hand_distance_diff
```

### Examples:

**Scenario 1: Both hands far (starting position)**
```
left_dist = 0.6m, right_dist = 0.6m
diff = |0.6 - 0.6| = 0
penalty = -2.0 × 0 = 0
```
No penalty, both hands equally far.

**Scenario 2: One hand close, one far (bad)**
```
left_dist = 0.05m, right_dist = 0.5m
diff = |0.05 - 0.5| = 0.45m
penalty = -2.0 × 0.45 = -0.9
```
Significant penalty for asymmetry!

**Scenario 3: Both hands close (goal)**
```
left_dist = 0.05m, right_dist = 0.06m
diff = |0.05 - 0.06| = 0.01m
penalty = -2.0 × 0.01 = -0.02
```
Minimal penalty, both hands reaching.

**Scenario 4: Both hands touching**
```
left_dist = 0.03m, right_dist = 0.04m
diff = |0.03 - 0.04| = 0.01m
penalty = -2.0 × 0.01 = -0.02
contact_reward = 10 × 2 = 20
both_hands_bonus = 50
total = 20 + 50 - 0.02 = +69.98
```
Huge reward!

---

## 🎓 Key Insights

### 1. **Local Optima Are Common in RL**
When agent finds a "good enough" strategy, it stops exploring better strategies. Must make better strategies MUCH more rewarding.

### 2. **Incremental Benefit Matters**
It's not about absolute reward values, but the **difference** between strategies:
- Old: +20 → +55 (difference: +35)
- New: +10 → +70 (difference: +60)

The larger difference motivates exploration.

### 3. **Penalties Can Guide Behavior**
Asymmetry penalty continuously pushes agent away from one-hand strategy, even before it discovers two-hand strategy.

### 4. **Reward Shaping Is Iterative**
- Phase 1: Got robot to reach piano (any hand)
- Phase 2: Encouraged both hands (but one hand was good enough)
- Phase 3: Force both hands (this fix)

---

## 🚀 What To Do Now

### If Training Fresh:
```bash
cd /home/solotech007/RoboGym/simulation
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --num_envs 2048 \
    --headless
```

### If Continuing Current Training:
The current policy has learned one-hand strategy. You have two options:

**Option A: Continue training** (may unlearn slowly)
- The new reward structure will gradually encourage two hands
- May take 500-1000 more epochs to shift behavior

**Option B: Restart training** (faster, recommended)
- Fresh policy will learn two-hand strategy directly
- Should see both hands reaching by epoch 500

---

## 📈 Expected Training Progress

### With New Reward Structure:

```
Reward
  |     
+6000|                      ___________  ← Both hands stable
     |                   .-'
+4000|            Both hands learning
     |          .-'
+2000|    One hand (transitioning)
     |  .-'
   0 |.'
     |'___________________________
      0   200  400  600  800  1000
               Epochs
```

**Milestones**:
- **0-200 epochs**: Learning to reach (one hand first)
- **200-500 epochs**: Discovering two-hand strategy (+70 reward compelling!)
- **500+ epochs**: Both hands consistently touching

---

## 🔧 Tuning If Needed

### If robot still uses one hand after 500 epochs:

**Increase both-hands bonus even more**:
```python
rew_scale_both_hands = 75.0  # Was 50.0
```

**Or increase asymmetry penalty**:
```python
rew_scale_hand_asymmetry = -5.0  # Was -2.0
```

### If robot struggles to reach at all:

**Reduce asymmetry penalty** (might be too harsh):
```python
rew_scale_hand_asymmetry = -1.0  # Was -2.0
```

---

## ✅ Success Metrics

After retraining, you should see:

1. **Reward**: +5000 to +7000 (higher than before due to both-hands bonus)
2. **Visual**: Both hands reaching and touching piano simultaneously
3. **TensorBoard**: 
   - `rewards/both_hands`: Should be high and frequent
   - `rewards/asymmetry`: Should be close to zero (symmetric)
   - `rewards/contact`: Consistent (both hands touching)

---

## 📚 Summary

| Change | Value | Reason |
|--------|-------|--------|
| **Contact reward** | 20.0 → **10.0** | Make single hand less appealing |
| **Both hands bonus** | 15.0 → **50.0** | Make coordination very compelling |
| **Asymmetry penalty** | N/A → **-2.0** | Continuously discourage one-hand strategy |

**Result**: Two-hand strategy is now **6x more rewarding** than one-hand strategy (+60 vs +10), with continuous penalty for asymmetry.

This should strongly guide the agent toward the desired both-hands behavior! 🎯🙌

