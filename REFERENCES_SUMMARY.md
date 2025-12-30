# References Summary

Quick guide to what references are available in which documents.

---

## Where to Find What

### 🎯 Quick Start References → `START_HERE.md`
**Best for**: Copy-paste commands to get started quickly

**Contains**:
- Primary Isaac Lab examples (Franka Cabinet, Cartpole, etc.)
- Quick copy commands for environment template
- Quick copy commands for training config
- Essential documentation links

**Read time**: 5 minutes

---

### 📖 Complete References → `RL_PIANO_REACH_PLAN.md`
**Best for**: Deep dive into all available resources

**Contains**:
- **Isaac Lab Examples** (with paths and what they're good for)
  - Manipulation tasks (Franka Cabinet, In-Hand)
  - Humanoid tasks
  - Basic examples (Cartpole)
  
- **Isaac Lab Core APIs** (with file paths)
  - Environment framework
  - Robot control (Articulation)
  - Sensors (ContactSensor)
  - Scene management
  
- **Unitree G1 Resources**
  - G1 configurations in Isaac Lab
  - Unitree Sim project in your workspace
  
- **RL Training Frameworks**
  - RL Games (recommended)
  - RSL-RL (alternative)
  - Stable-Baselines3 (alternative)
  
- **Learning Resources**
  - Isaac Lab tutorials
  - RL theory (OpenAI Spinning Up)
  - Papers (PPO, Isaac Gym)
  
- **Command-Line Tools**
  - Training commands
  - Inference commands
  - Debug tools
  
- **Related Projects**
  - Your Unitree Sim project reference

**Read time**: 30 minutes

---

### 🗺️ Implementation References → `RL_IMPLEMENTATION_ROADMAP.md`
**Best for**: References while coding

**Contains**:
- Primary examples to study (ranked by relevance)
- Key files to reference (specific file paths)
- Quick commands (copy, search, debug)
- Documentation links
- Where to get help if stuck

**Read time**: 10 minutes

---

### 📚 Project Overview → `README.md`
**Best for**: Sharing with others, project navigation

**Contains**:
- Key Isaac Lab examples (organized by category)
- Isaac Lab documentation links
- Unitree G1 resources
- RL training frameworks
- Learning resources
- Command reference
- Quick links to all project documents

**Read time**: 15 minutes

---

## Primary References by Use Case

### "I want to start coding NOW"
→ **`START_HERE.md`** → "Key References" section
- Copy franka_cabinet_env.py as template
- Copy rl_games_ppo_cfg.yaml for training

### "I need to understand the full picture"
→ **`RL_PIANO_REACH_PLAN.md`** → "Resources & References" section
- Complete list of all examples
- All APIs with file paths
- All documentation links

### "I'm stuck on implementation"
→ **`RL_IMPLEMENTATION_ROADMAP.md`** → "Getting Help" section
- Common issues
- Where to look for specific code patterns
- Search commands

### "I want to learn RL concepts"
→ **`RL_PIANO_REACH_PLAN.md`** → "Learning Resources"
- OpenAI Spinning Up (RL theory)
- Isaac Lab tutorials (environment creation)
- Papers (PPO, Isaac Gym)

### "I need training config details"
→ **`RL_PIANO_REACH_PLAN.md`** → "Training Configuration" section
- References to actual Isaac Lab configs used
- Franka Cabinet (primary)
- Humanoid (secondary)
- Cartpole (baseline)
- Explanation of why each parameter was chosen

---

## Key Isaac Lab Files Referenced

### Direct RL Environment Examples (in order of relevance)

1. **`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/franka_cabinet/`**
   - `franka_cabinet_env.py` - Environment class
   - `agents/rl_games_ppo_cfg.yaml` - Training config
   - **Use as**: PRIMARY TEMPLATE (copy this!)

2. **`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/`**
   - `cartpole_env.py` - Simple structure
   - `agents/rl_games_ppo_cfg.yaml` - Basic config
   - **Use as**: Understanding basics

3. **`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/inhand_manipulation/`**
   - `inhand_manipulation_env.py` - Contact rewards
   - **Use as**: Reference for contact sensing

4. **`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid/`**
   - `humanoid_env.py` - Humanoid control
   - `agents/rl_games_ppo_cfg.yaml` - Humanoid config
   - **Use as**: Network sizing reference

### Core API Files

- **DirectRLEnv**: `IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py`
- **Articulation**: `IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py`
- **ContactSensor**: `IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor.py`

---

## External Documentation

### Essential
- **Isaac Lab**: https://isaac-sim.github.io/IsaacLab/
- **RL Games**: https://github.com/Denys88/rl_games

### Helpful
- **OpenAI Spinning Up**: https://spinningup.openai.com/ (RL theory)
- **RSL-RL**: https://github.com/leggedrobotics/rsl_rl (alternative framework)
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/ (alternative framework)

### Deep Dives
- **PPO Paper**: https://arxiv.org/abs/1707.06347
- **Isaac Gym Paper**: https://arxiv.org/abs/2108.10470
- **Isaac Lab GitHub**: https://github.com/isaac-sim/IsaacLab

---

## Quick Command Reference

### Copy Templates
```bash
# Environment class
cp IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/franka_cabinet/franka_cabinet_env.py \
   g1-piano-play/envs/g1_piano_reach_env.py

# Training config
cp IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/franka_cabinet/agents/rl_games_ppo_cfg.yaml \
   g1-piano-play/envs/agents/rl_games_ppo_cfg.yaml
```

### Search for API Usage
```bash
# Find how body positions are accessed
grep -r "body_pos_w" IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/

# Find contact sensor usage
grep -r "ContactSensor" IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/

# List all direct RL examples
ls IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/
```

### Training & Inference
```bash
# Train
./isaaclab.sh -p scripts/train.py --task Isaac-Piano-Reach-G1-v0 --num_envs 1024 --headless

# Play
./isaaclab.sh -p scripts/play.py --task Isaac-Piano-Reach-G1-v0 --checkpoint model.pth

# TensorBoard
tensorboard --logdir runs/
```

---

## References Added to Each Document

### ✅ RL_PIANO_REACH_PLAN.md
- [x] Isaac Lab examples (manipulation, humanoid, basic)
- [x] Core APIs with file paths
- [x] Unitree G1 resources
- [x] RL training frameworks
- [x] Learning resources (tutorials, theory, papers)
- [x] Command-line tools
- [x] Helper scripts
- [x] Related projects in workspace
- [x] Training config references (with rationale)

### ✅ RL_IMPLEMENTATION_ROADMAP.md
- [x] Primary examples (ranked)
- [x] Key files to reference
- [x] Quick commands (copy, search)
- [x] Documentation links
- [x] Getting help section

### ✅ README.md
- [x] Key examples (organized by category)
- [x] Documentation links
- [x] Unitree G1 resources
- [x] Training frameworks
- [x] Learning resources
- [x] Command reference
- [x] Quick links to project docs

### ✅ START_HERE.md
- [x] Primary examples
- [x] Quick copy commands
- [x] Essential documentation links
- [x] Reference to detailed docs

### ✅ APPROACH_COMPARISON.md
- [x] References in context (already had some)

---

## Using This Summary

**Tomorrow morning workflow**:
1. Open `START_HERE.md` → Quick reference
2. Copy franka_cabinet as template
3. Start Day 1 tasks from roadmap
4. Keep `RL_PIANO_REACH_PLAN.md` open for detailed reference
5. Use `RL_IMPLEMENTATION_ROADMAP.md` for step-by-step guidance

**When stuck**:
1. Check "Common Issues" in roadmap
2. Search for similar code in franka_cabinet
3. Grep for API usage: `grep -r "body_pos_w" IsaacLab/.../direct/`
4. Read relevant section in `RL_PIANO_REACH_PLAN.md`

**For others reviewing your project**:
1. Point them to `README.md` first
2. Then `START_HERE.md` for orientation
3. Then `APPROACH_COMPARISON.md` for rationale

---

## Summary

All documents now have comprehensive references! ✅

**Most detailed**: `RL_PIANO_REACH_PLAN.md` (500+ lines of references)  
**Most practical**: `START_HERE.md` + `RL_IMPLEMENTATION_ROADMAP.md`  
**Most accessible**: `README.md`

You're ready to start implementing! 🚀

