# G1 Piano Reaching - Quick Start Guide

## ✅ Setup Complete!

The RL environment is ready to use. Here's how to get started.

---

## 📁 Project Structure

```
g1-piano-play/
├── envs/
│   ├── __init__.py                          # Gym registration
│   ├── g1_piano_reach_env_cfg.py           # Environment configuration
│   ├── g1_piano_reach_env.py               # Environment implementation
│   └── agents/
│       ├── __init__.py
│       └── rl_games_ppo_cfg.yaml           # Training hyperparameters
├── scripts/
│   ├── test_env.py                         # Test environment setup
│   ├── train.py                            # Training script
│   └── play.py                             # Inference script
└── Documentation/
    ├── README.md                           # Full project overview
    ├── RL_PIANO_REACH_PLAN.md             # Complete technical plan
    ├── RL_IMPLEMENTATION_ROADMAP.md        # Implementation guide
    └── QUICKSTART.md                       # This file
```

---

## 🚀 Step-by-Step Usage

### Step 1: Test Environment (IMPORTANT - Do this first!)

Before training, verify the environment works:

```bash
cd /home/solotech007/RoboGym/simulation

# Test with visualization (16 envs, fast)
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/test_env.py --num_envs 16

# Test headless (faster, for checking setup)
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/test_env.py --num_envs 16 --headless
```

**What to check:**
- ✅ Environment creates without errors
- ✅ Observation shape is (16, 36)
- ✅ Robot appears sitting on bench
- ✅ Random actions work
- ✅ Episodes reset correctly

If test passes, proceed to training!

---

### Step 2: Train Policy

```bash
cd /home/solotech007/RoboGym/simulation

# Basic training (1024 envs, with visualization)
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --num_envs 1024

# Headless training (faster, recommended)
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --num_envs 1024 \
    --headless

# With video recording (slower, for debugging)
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --num_envs 256 \
    --video
```

**Training will:**
- Run for 2000 epochs (configurable)
- Save checkpoints every 100 epochs to `logs/rl_games/Isaac-Piano-Reach-G1-v0/`
- Log to TensorBoard in `runs/`

**Monitor progress:**
```bash
# In another terminal
tensorboard --logdir logs/rl_games/Isaac-Piano-Reach-G1-v0/
# Open browser to http://localhost:6006
```

---

### Step 3: Play Trained Policy

After training (or when checkpoint exists):

```bash
cd /home/solotech007/RoboGym/simulation

# Play with single environment
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/play.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --checkpoint logs/rl_games/Isaac-Piano-Reach-G1-v0/nn/best_checkpoint.pth \
    --num_envs 1

# Play with video recording
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/play.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --checkpoint logs/rl_games/Isaac-Piano-Reach-G1-v0/nn/best_checkpoint.pth \
    --num_envs 1 \
    --video
```

Videos saved to: `videos/play/Isaac-Piano-Reach-G1-v0/`

---

## 🔧 Configuration

### Environment Parameters

Edit `envs/g1_piano_reach_env_cfg.py`:

```python
# Number of parallel environments
scene.num_envs = 1024  # More = faster training, needs more GPU memory

# Episode length
episode_length_s = 10.0  # Seconds per episode

# Action scaling
action_scale = 0.5  # Lower = smoother but slower motion

# Reward scales (tune these!)
rew_scale_reaching = 2.0     # Distance to piano
rew_scale_contact = 10.0     # Touch bonus
rew_scale_both_hands = 5.0   # Both hands bonus
```

### Training Hyperparameters

Edit `envs/agents/rl_games_ppo_cfg.yaml`:

```yaml
# Network size
mlp.units: [256, 128, 64]  # Larger = more capacity

# Learning rate
learning_rate: 3e-4  # Lower = more stable, slower

# Training length
max_epochs: 2000  # More = longer training

# Batch sizes
horizon_length: 16      # Steps per rollout
minibatch_size: 8192    # Samples per update
```

---

## 📊 Expected Results

### Phase 1 Success Criteria:

After ~500-1000 epochs of training:
- ✅ Hands move toward piano targets
- ✅ Both hands reach piano >50% of episodes
- ✅ Smooth, controlled motion (no jerky movements)
- ✅ Robot stays seated (doesn't fall)

### Reward Progression:

```
Epoch    0-100:   Reward ~ -20 to -10  (learning to move arms)
Epoch  100-300:   Reward ~ -10 to 0    (getting closer to piano)
Epoch  300-500:   Reward ~ 0 to +10    (touching piano sometimes)
Epoch  500-1000:  Reward ~ +10 to +20  (consistent touching)
Epoch 1000+:      Reward ~ +20 to +30  (both hands consistently)
```

---

## 🐛 Troubleshooting

### Issue: Import Error

```
ImportError: No module named 'envs'
```

**Fix:** Run from correct directory:
```bash
cd /home/solotech007/RoboGym/simulation
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/test_env.py
```

### Issue: Environment Creation Fails

```
[ERROR]: Could not find hand bodies
```

**Fix:** Robot model may have different body names. Check output and update `_get_hand_body_indices()` in `envs/g1_piano_reach_env.py`.

### Issue: Training Not Improving

**Possible fixes:**
1. Lower learning rate: `learning_rate: 1e-4`
2. Increase reward scales: `rew_scale_reaching = 5.0`
3. Increase network size: `units: [512, 256, 128]`
4. Check if robot is falling (terminating early)

### Issue: Robot Falls Immediately

**Possible fixes:**
1. Check sitting pose in `_get_sitting_pose()`
2. Increase physics solver iterations in config
3. Verify bench height matches robot position

---

## 📈 Monitoring Training

### TensorBoard Metrics

Key metrics to watch:
- **`rewards/iter`**: Should increase over time
- **`losses/policy_loss`**: Should decrease and stabilize
- **`info/episode_length`**: Should increase (not terminating early)
- **`info/success_rate`**: Custom metric (if added)

### Console Output

```
Epoch: 500
reward: 15.234
length: 600
fps: 12000
```

- **reward**: Average reward (target: +20 to +30)
- **length**: Average episode length (target: 600 = full episode)
- **fps**: Frames per second (should be >10k with GPU)

---

## 🎯 Next Steps

### After Phase 1 Works:

1. **Tune Rewards**
   - Adjust scales for better performance
   - Add success bonus when both hands touch

2. **Add Contact Sensor (Option B)**
   - Uncomment ContactSensor code
   - Set `contact_detection_method = "sensor"`
   - More accurate force detection

3. **Phase 2: Piano Keys**
   - Replace simple piano with detailed keyboard
   - Add key-specific targets
   - Train to press specific keys

### Upgrade Observation Space (if needed):

If robot tips/falls, add stability sensors:

```python
# In _get_observations():
projected_gravity = self.robot.data.projected_gravity_b  # 3
root_ang_vel = self.robot.data.root_ang_vel_b  # 3

obs = torch.cat([
    arm_pos,
    arm_vel,
    left_hand_to_target,
    right_hand_to_target,
    projected_gravity,   # NEW
    root_ang_vel,        # NEW
    prev_actions,
], dim=-1)  # Total: 42 dims
```

Update `observation_space = 42` in config.

---

## 📚 Full Documentation

- **Technical Plan**: `RL_PIANO_REACH_PLAN.md`
- **Implementation Guide**: `RL_IMPLEMENTATION_ROADMAP.md`
- **Project Overview**: `README.md`
- **RL vs Animation**: `APPROACH_COMPARISON.md`

---

## ✅ Quick Checklist

Before training:
- [ ] Tested environment with `test_env.py`
- [ ] Environment creates without errors
- [ ] Robot sits on bench correctly
- [ ] Observations have correct shape (36 dims)
- [ ] Random actions work

During training:
- [ ] TensorBoard shows increasing rewards
- [ ] Episode lengths are reasonable (>300 steps)
- [ ] No NaN/Inf in losses
- [ ] Checkpoints are being saved

After training:
- [ ] Can load and play trained policy
- [ ] Hands move toward piano
- [ ] Success rate >50%

---

## 🚀 You're Ready!

Start with:
```bash
cd /home/solotech007/RoboGym/simulation
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/test_env.py --num_envs 16
```

Good luck! 🎹🤖✨

