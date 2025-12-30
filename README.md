# G1 Piano Reach - RL Training Environment

An Isaac Lab reinforcement learning environment where a Unitree G1 humanoid robot learns to reach and touch piano keys while sitting on a bench.

## 📋 Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Training](#training)
- [Testing & Evaluation](#testing--evaluation)
- [Configuration](#configuration)
- [Monitoring Progress](#monitoring-progress)
- [Troubleshooting](#troubleshooting)

---

## Overview

**Task**: Train a G1 robot (sitting in a fixed position) to reach forward and touch piano keys with both hands.

**Key Features**:
- 2048 parallel environments for fast training
- 10 DOF action space (arm joints only, legs fixed in sitting pose)
- 36-dimensional observation space
- Distance-based reward with contact detection
- PPO algorithm via RL Games

**Phase 1 Goal**: Successfully touch the piano with both hands  
**Phase 2 Goal** (future): Press specific keys on command

---

## Quick Start

### 1. Activate Virtual Environment
```bash
cd /home/solotech007/RoboGym/simulation
source env_isaaclab/bin/activate
```

### 2. Test the Environment (Recommended First Step)
```bash
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/test_env.py --num_envs 16
```

This will spawn 16 environments with random actions to verify everything works.

### 3. Start Training
```bash
# Headless mode (no visualization, faster)
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --headless

# OR with visualization (to watch robots learn)
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py \
    --task Isaac-Piano-Reach-G1-v0
```

---

## Training

### Basic Training Commands

**Default Training** (2048 envs, headless):
```bash
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --headless
```

**Custom Number of Environments**:
```bash
# If you have limited GPU memory
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --num_envs 512 \
    --headless
```

**With Video Recording**:
```bash
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --video \
    --video_length 200 \
    --video_interval 500
```

### Resume Training from Checkpoint

```bash
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --resume \
    --load_run <run_folder_name>
```

Example:
```bash
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --resume \
    --load_run 2024-01-15_10-30-45
```

---

## Testing & Evaluation

### Test Environment with Random Actions
```bash
# Test with 16 environments
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/test_env.py --num_envs 16

# Test with 100 episodes
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/test_env.py \
    --num_envs 16 \
    --num_episodes 100
```

### Play Trained Policy (Inference)
```bash
# Load and run a trained policy
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/play.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --checkpoint /path/to/checkpoint.pth \
    --num_envs 16
```

### Evaluate Policy Performance
```bash
# Run evaluation with video recording
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/play.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --checkpoint logs/rl_games/g1_piano_reach/nn/last_checkpoint.pth \
    --num_envs 64 \
    --video \
    --video_length 500
```

---

## Configuration

### Environment Configuration
Edit `envs/g1_piano_reach_env_cfg.py` to modify:

**Number of Environments**:
```python
scene: InteractiveSceneCfg = InteractiveSceneCfg(
    num_envs=2048,  # Change this
    env_spacing=3.0,
    ...
)
```

**Reward Scales**:
```python
# Tune these for different learning behaviors
rew_scale_reaching = 2.0      # Reward for getting closer to piano
rew_scale_contact = 10.0      # Bonus for touching piano
rew_scale_action_penalty = -0.01  # Penalize large actions
rew_scale_fall_penalty = -100.0   # Penalize falling
```

**Episode Length**:
```python
episode_length_s = 10.0  # 10 seconds per episode
```

### Training Configuration
Edit `envs/agents/rl_games_ppo_cfg.yaml` to modify:

**Network Architecture**:
```yaml
network:
  mlp:
    units: [256, 128, 64]  # Hidden layer sizes
```

**Learning Rate**:
```yaml
learning_rate: 3e-4  # Adjust for faster/slower learning
```

**Training Duration**:
```yaml
max_epochs: 2000  # Total training epochs
```

---

## Monitoring Progress

### TensorBoard (Real-time Metrics)
**Open a NEW terminal** while training is running:
```bash
cd /home/solotech007/RoboGym/simulation
source env_isaaclab/bin/activate
tensorboard --logdir logs/rl_games/Isaac-Piano-Reach-G1-v0/
```

Then open in your browser: **http://localhost:6006**

**Key Metrics to Watch**:

| Metric | What It Means | Good Sign |
|--------|---------------|-----------|
| `rewards/iter` | Average reward per iteration | Increasing trend |
| `rewards/episode` | Average episode reward | Trending upward |
| `episode_lengths/iter` | Episode duration | Reaching max (1200 steps) |
| `losses/a_loss` | Actor (policy) loss | Stabilizing |
| `losses/c_loss` | Critic (value) loss | Decreasing |
| `info/lr` | Learning rate | Adaptive schedule |

**Training Phases**:
- **0-100 epochs**: Low/negative rewards (exploration)
- **100-500 epochs**: Rewards climbing (learning)
- **500+ epochs**: Rewards plateau (convergence)

### Test Checkpoints During Training

Checkpoints are auto-saved every **100 epochs** to:
```
logs/rl_games/Isaac-Piano-Reach-G1-v0/nn/
├── G1PianoReach_ep_50.pth
├── G1PianoReach_ep_100.pth
├── G1PianoReach_ep_200.pth
└── last_checkpoint.pth  # Most recent
```

**Test a checkpoint** (in a new terminal while training runs):
```bash
cd /home/solotech007/RoboGym/simulation

# Test latest checkpoint with 4 environments
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/test_checkpoint.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --checkpoint logs/rl_games/Isaac-Piano-Reach-G1-v0/nn/last_checkpoint.pth \
    --num_envs 4 \
    --num_episodes 10

# Test specific epoch (e.g., epoch 500)
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/test_checkpoint.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --checkpoint logs/rl_games/Isaac-Piano-Reach-G1-v0/nn/G1PianoReach_ep_500.pth \
    --num_envs 16 \
    --num_episodes 20
```

**What to expect**:
- Early checkpoints (ep 50-100): Random/poor reaching behavior
- Mid training (ep 200-500): Arms moving toward piano
- Late training (ep 1000+): Smooth, consistent reaching and touching

### Watch Training Output in Real-time

Training prints progress every epoch:
```
Ep #500 | Mean Reward: 12.45 | Mean Length: 850.2 | FPS: 25430
         | Actor Loss: 0.0234 | Critic Loss: 0.1234 | Entropy: 0.000
```

**To monitor from another terminal**:
```bash
# Watch the latest log file
tail -f logs/rl_games/Isaac-Piano-Reach-G1-v0/*/log.txt
```

### Log Files
Training logs and checkpoints are saved to:
```
logs/rl_games/Isaac-Piano-Reach-G1-v0/
├── <timestamp>/
│   ├── nn/                  # Model checkpoints (.pth files)
│   ├── summaries/           # TensorBoard event files
│   ├── log.txt              # Training output log
│   └── config.yaml          # Full training configuration
```

### Checkpoints
Model checkpoints saved every 50 epochs:
```
logs/rl_games/g1_piano_reach/nn/
├── G1PianoReach_ep_50.pth
├── G1PianoReach_ep_100.pth
├── ...
└── last_checkpoint.pth      # Most recent
```

---

## Troubleshooting

### Out of Memory (OOM)
**Symptoms**: Training crashes with CUDA OOM error

**Solutions**:
1. Reduce number of environments:
   ```bash
   ./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py \
       --task Isaac-Piano-Reach-G1-v0 \
       --num_envs 512 \
       --headless
   ```

2. Reduce batch size in `envs/agents/rl_games_ppo_cfg.yaml`:
   ```yaml
   minibatch_size: 4096  # Down from 8192
   ```

### Robot Falling Immediately
**Symptoms**: All robots fall down at start

**Check**:
1. Initial joint positions are within limits
2. Robot spawn height is correct (`z=0.60`)
3. Sitting pose is stable

**Fix**: The current config has been tested and should work. If issues persist, check `envs/g1_piano_reach_env_cfg.py` initial joint positions.

### Table/Piano Not Visible
**Symptoms**: Objects missing or buried

**Solution**: Already fixed! Using procedural primitives that clone reliably. If issues persist:
```bash
# Test spawning with fewer envs
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/test_env.py --num_envs 4
```

### Import Errors
**Symptoms**: `ModuleNotFoundError` for custom environment

**Solution**:
```bash
# Ensure you're in the simulation directory
cd /home/solotech007/RoboGym/simulation

# Activate the correct environment
source env_isaaclab/bin/activate

# Run through isaaclab.sh launcher (not direct python)
./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py ...
```

### Slow Training
**Symptoms**: Low FPS during training

**Tips**:
1. **Use headless mode**: Add `--headless` flag
2. **Increase physics substeps**: May reduce stability but faster
3. **GPU utilization**: Check with `nvidia-smi` - should be 90%+
4. **Reduce render interval**: In config, increase `decimation`

---

## Project Structure

```
g1-piano-play/
├── envs/
│   ├── __init__.py                    # Gym registration
│   ├── g1_piano_reach_env.py          # Main environment class
│   ├── g1_piano_reach_env_cfg.py      # Environment configuration
│   └── agents/
│       └── rl_games_ppo_cfg.yaml      # PPO training config
├── scripts/
│   ├── train.py                       # Training script
│   ├── test_env.py                    # Testing script
│   └── play.py                        # Inference script
├── tutorials/
│   └── 00_sim/
│       └── g1-piano-play.py           # Original animation demo
├── RL_PIANO_REACH_PLAN.md             # Technical design doc
├── RL_IMPLEMENTATION_ROADMAP.md       # Implementation guide
├── IMPLEMENTATION_SUMMARY.md          # Code overview
├── QUICKSTART.md                      # Quick reference
├── RECENT_FIXES.md                    # Bug fixes log
└── README.md                          # This file
```

---

## Expected Training Timeline

### Phase 1: Initial Exploration (0-200 epochs)
- Robots learn basic arm movement
- Random flailing, exploring action space
- Low rewards (~0-5)

### Phase 2: Directed Reaching (200-500 epochs)
- Robots start moving arms toward piano
- Rewards increase (~5-15)
- Occasional successful touches

### Phase 3: Consistent Contact (500-1000 epochs)
- Most robots reliably touch piano
- Rewards plateau (~15-25)
- Learning to optimize approach

### Phase 4: Fine-tuning (1000-2000 epochs)
- Smooth, efficient reaching motions
- High success rate (>90%)
- Stable policy

**Typical Training Time**: 
- On RTX 3090/4090: ~2-4 hours for 2000 epochs
- On A100: ~1-2 hours

---

## Next Steps (Phase 2)

Once the robot reliably touches the piano, Phase 2 will involve:

1. **Replace piano with articulated keyboard** (individual keys)
2. **Add key-specific targets** to observation space
3. **Reward for pressing correct keys** based on input
4. **Implement ContactSensor** for physical key press detection
5. **Train piano-playing policy** with note sequences

---

## Citation

Built with [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) and trained using [RL Games](https://github.com/Denys88/rl_games).

---

## License

BSD-3-Clause (follows Isaac Lab licensing)

---

## Support

For issues or questions:
1. Check [TROUBLESHOOTING](#troubleshooting) section
2. Review `RECENT_FIXES.md` for known issues
3. Check Isaac Lab documentation: https://isaac-sim.github.io/IsaacLab/
