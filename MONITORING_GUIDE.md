# G1 Piano Reach - Training Monitoring Guide

Quick reference for monitoring and testing your training in real-time.

---

## 🔍 Real-time Monitoring (3 Ways)

### 1. TensorBoard (Best for Visualizing Progress)

**Start in new terminal:**
```bash
cd /home/solotech007/RoboGym/simulation
source env_isaaclab/bin/activate
tensorboard --logdir logs/rl_games/Isaac-Piano-Reach-G1-v0/
```

**Open browser:** http://localhost:6006

**Key Metrics:**
| Metric | What to Watch |
|--------|---------------|
| `rewards/iter` | Should increase steadily |
| `episode_lengths/iter` | Should reach ~1200 (max episode length) |
| `losses/c_loss` | Should decrease over time |
| `info/lr` | Tracks learning rate schedule |

---

### 2. Watch Training Logs

**In new terminal:**
```bash
cd /home/solotech007/RoboGym/simulation
tail -f logs/rl_games/Isaac-Piano-Reach-G1-v0/*/log.txt
```

**Look for:**
```
Ep #100 | Mean Reward: 5.23 | Mean Length: 456.2 | FPS: 28430
Ep #200 | Mean Reward: 12.45 | Mean Length: 850.2 | FPS: 27890
Ep #500 | Mean Reward: 18.67 | Mean Length: 1150.3 | FPS: 26340
```

---

### 3. Training Console Output

The training terminal shows:
- Current epoch number
- Mean reward (should increase)
- Episode length (should increase)
- Training FPS (steps per second)
- Loss values

---

## 🎮 Test Checkpoints Mid-Training

### Checkpoint Locations

Checkpoints saved every **100 epochs** (or as configured):
```
logs/rl_games/Isaac-Piano-Reach-G1-v0/nn/
├── G1PianoReach_ep_50.pth         # First checkpoint
├── G1PianoReach_ep_100.pth
├── G1PianoReach_ep_200.pth
├── ...
└── last_checkpoint.pth            # Most recent (updates continuously)
```

### Test Latest Checkpoint

**In new terminal** (while training continues):
```bash
cd /home/solotech007/RoboGym/simulation

./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/test_checkpoint.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --checkpoint logs/rl_games/Isaac-Piano-Reach-G1-v0/nn/last_checkpoint.pth \
    --num_envs 4 \
    --num_episodes 10
```

**Output:**
```
[Episode   1] Reward:    12.45 | Length:  850 steps
[Episode   2] Reward:    14.23 | Length:  920 steps
...
================================================================================
RESULTS (10 episodes)
================================================================================
Average Reward:     13.56
Average Length:    875.20 steps
================================================================================
```

### Test Specific Epoch

```bash
# Test checkpoint from epoch 500
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/test_checkpoint.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --checkpoint logs/rl_games/Isaac-Piano-Reach-G1-v0/nn/G1PianoReach_ep_500.pth \
    --num_envs 16 \
    --num_episodes 20
```

---

## 📊 What Good Training Looks Like

### Phase 1: Early Exploration (Epochs 0-100)
- **Rewards**: Low or negative (-5 to 5)
- **Episode Lengths**: Short (200-400 steps)
- **Behavior**: Random arm movements, falling

### Phase 2: Learning (Epochs 100-500)
- **Rewards**: Climbing (5 to 15)
- **Episode Lengths**: Increasing (400-800 steps)
- **Behavior**: Arms moving toward piano, occasional contact

### Phase 3: Refinement (Epochs 500-1000)
- **Rewards**: Steady increase (15 to 25)
- **Episode Lengths**: Near max (800-1100 steps)
- **Behavior**: Consistent reaching, frequent piano contact

### Phase 4: Convergence (Epochs 1000+)
- **Rewards**: Plateau (20-30)
- **Episode Lengths**: Max (1100-1200 steps)
- **Behavior**: Smooth reaching, reliable piano touching

---

## ⚠️ Warning Signs

### Training is Not Working If:

1. **Rewards stay flat or negative after 200 epochs**
   - Check reward scales in config
   - Verify robot spawns correctly
   - Check target positions are reachable

2. **Episode lengths don't increase**
   - Robot might be falling immediately
   - Check initial pose stability
   - Verify termination conditions

3. **Loss values explode (NaN or very large)**
   - Reduce learning rate
   - Check observation normalization
   - Reduce gradient norm clipping

4. **Training FPS drops below 10,000**
   - Reduce num_envs
   - Check GPU memory usage
   - Close unnecessary applications

---

## 🛑 Stopping and Resuming Training

### Graceful Stop
Press `Ctrl+C` in the training terminal. It will:
- Save the current checkpoint
- Close simulation cleanly

### Resume Training
```bash
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --checkpoint logs/rl_games/Isaac-Piano-Reach-G1-v0/nn/last_checkpoint.pth \
    --headless
```

---

## 📈 Typical Training Timeline

**On RTX 3090/4090 with 2048 environments:**

| Epochs | Time | Expected Behavior |
|--------|------|-------------------|
| 50 | ~2 min | Random movements |
| 100 | ~4 min | Some arm coordination |
| 200 | ~8 min | Arms reaching forward |
| 500 | ~20 min | Occasional piano touches |
| 1000 | ~40 min | Consistent piano contact |
| 2000 | ~80 min | Smooth, reliable reaching |

**Total training time: ~1.5-2 hours for full convergence**

---

## 🎯 Success Criteria (When to Stop Training)

Training is successful when:

✅ **Average reward > 20** for 100+ consecutive epochs  
✅ **Episode length > 1000** steps consistently  
✅ **Contact reward > 0** in most episodes  
✅ **Both hands reach piano** in test runs  
✅ **Behavior is smooth** and repeatable

You can stop training and use the checkpoint for Phase 2!

---

## 💡 Pro Tips

1. **Compare checkpoints**: Test checkpoints from different epochs to see improvement
2. **Watch videos**: Enable `--video` during testing to record behavior
3. **Multiple tests**: Run 20-50 episodes for reliable statistics
4. **GPU monitoring**: Use `nvidia-smi -l 1` to watch GPU utilization
5. **Backup checkpoints**: Copy promising checkpoints to a safe location

---

## 🆘 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Training very slow | Add `--headless` flag |
| Out of memory | Reduce `--num_envs` (try 1024 or 512) |
| TensorBoard not showing data | Wait 1-2 minutes, refresh browser |
| Checkpoint test fails | Check checkpoint path, ensure training created it |
| Can't see robots | Remove `--headless` from training command |

---

**Happy Training! 🚀🎹**

