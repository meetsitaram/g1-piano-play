# Dynamic Batch Size Configuration

## 🎯 Problem Solved

**Before:** Changing `--num_envs` between runs would cause this error:
```
AssertionError: assert(self.batch_size % self.minibatch_size == 0)
```

**After:** `minibatch_size` is now **automatically calculated** based on `num_envs`!

---

## ✅ How It Works

### The Math:
```python
batch_size = num_envs × horizon_length
```

**Requirement:** `batch_size % minibatch_size == 0`

### Examples:

| num_envs | horizon_length | batch_size | minibatch_size (auto) | Valid? |
|----------|----------------|------------|-----------------------|--------|
| 2048 | 16 | 32768 | 8192 (batch/4) | ✅ |
| 1024 | 16 | 16384 | 4096 (batch/4) | ✅ |
| 256 | 16 | 4096 | 1024 (batch/4) | ✅ |
| 16 | 16 | 256 | 128 (batch/2) | ✅ |
| 4 | 16 | 64 | 64 (batch/1) | ✅ |

---

## 🔧 Implementation

### In `train.py`:

The script now automatically calculates `minibatch_size` before training starts:

```python
# Calculate batch_size
num_envs = env.unwrapped.num_envs
horizon_length = rl_games_cfg_omega.params.config.horizon_length
batch_size = num_envs * horizon_length

# Auto-calculate minibatch_size
if batch_size >= 8192:
    minibatch_size = max(batch_size // 4, 8192)
elif batch_size >= 1024:
    minibatch_size = batch_size // 4
elif batch_size >= 256:
    minibatch_size = max(batch_size // 2, 64)
else:
    minibatch_size = batch_size

# Ensure even division
while batch_size % minibatch_size != 0:
    minibatch_size = minibatch_size // 2

# Update config
rl_games_cfg_omega.params.config.minibatch_size = minibatch_size
```

### Strategy:

1. **Large batches (≥8192)**: Use batch/4 (for better gradient estimates)
2. **Medium batches (1024-8191)**: Use batch/4
3. **Small batches (256-1023)**: Use batch/2 or 64, whichever is larger
4. **Very small batches (<256)**: Use full batch
5. **Fallback**: Ensure divisibility by repeatedly dividing by 2

---

## 📊 Training Output

When you run training, you'll see:

```
[INFO]: Batch configuration:
         num_envs = 16
         horizon_length = 16
         batch_size = 256
         minibatch_size = 128 (auto-calculated)
         mini_epochs = 8
```

This confirms the automatic calculation worked correctly!

---

## 🚀 Usage

### No changes needed! Just run with any num_envs:

```bash
# Small test run (16 envs)
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --num_envs 16

# Medium run (256 envs)
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --num_envs 256

# Full run (2048 envs)
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --num_envs 2048 \
    --headless
```

**All will work automatically!** ✅

---

## 🎯 Why minibatch_size Matters

### In PPO Training:

1. **Collect data**: Sample `batch_size` transitions from environment
2. **Split into minibatches**: Divide batch into smaller chunks
3. **Update policy**: Train on each minibatch multiple times (`mini_epochs`)

### Impact of minibatch_size:

| Size | Training Speed | Gradient Quality | GPU Usage |
|------|----------------|------------------|-----------|
| Too large | Slow updates | Better (less noise) | High |
| Too small | Fast updates | Worse (more noise) | Low |
| Balanced | Good | Good | Optimal |

**Our auto-calculation aims for batch/4 ratio**, which is a good balance used in many Isaac Lab examples.

---

## 🔧 Manual Override (Optional)

If you want to manually set `minibatch_size` for experimentation:

### Option 1: Modify YAML before training
```yaml
# In envs/agents/rl_games_ppo_cfg.yaml
minibatch_size: 4096  # Your custom value
```

Then comment out the auto-calculation in `train.py`.

### Option 2: Pass via CLI (requires script modification)
Add a `--minibatch_size` argument to `train.py` if needed.

---

## 📚 RL Games Requirements

From `rl_games/common/a2c_common.py`:

```python
# Line 256
assert(self.batch_size % self.minibatch_size == 0)
```

**This assertion is why we need dynamic calculation!**

Additional constraints:
- `minibatch_size > 0`
- `minibatch_size <= batch_size`
- `batch_size % minibatch_size == 0` ← **CRITICAL**

---

## ✅ Benefits

1. ✅ **No more assertion errors** when changing `--num_envs`
2. ✅ **Automatic optimization** for different environment counts
3. ✅ **No manual config edits** needed between runs
4. ✅ **Clear logging** shows what was calculated
5. ✅ **Fallback handling** for edge cases

---

## 🎓 Technical Notes

### Why divide by 4?

Common practice in RL training:
- Too many minibatches (small minibatch_size) → noisy gradients
- Too few minibatches (large minibatch_size) → expensive updates
- **batch/4 to batch/8** is a sweet spot

### Why check divisibility?

PPO needs to:
1. Split batch into minibatches **evenly**
2. Ensure all data is used **exactly once** per epoch
3. Avoid **remainders** that would be discarded

If `batch_size % minibatch_size != 0`, some data would be left over!

---

## 🚀 Result

You can now **seamlessly switch** between:
- Visual testing (16 envs)
- Quick iteration (256 envs)
- Full training (2048 envs)

**Without ever touching the config file!** 🎉

