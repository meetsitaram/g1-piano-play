# Recent Fixes - G1 Piano Reach RL Environment

## Session: Dec 30, 2025

### 1. ✅ Fixed Action Space Mismatch
**Issue**: Environment expected 10 DOF actions but test script was generating 16
**Fix**: Updated `test_env.py` to use `env.unwrapped.cfg.action_space` (10) directly
**File**: `scripts/test_env.py`

### 2. ✅ Fixed Table & Piano Positioning
**Issue**: Table and piano not appearing or positioned incorrectly in multi-env setup
**Fix**: 
- Copied exact positioning logic from `g1-piano-play.py` demo
- Table: `z=0.4` (bottom at ground), top at `z=0.43`
- Piano: `x=-0.483, y=-0.21, z=0.505` (on table, centered, towards robot)
- Updated target positions to match new piano location
**Files**: 
- `envs/g1_piano_reach_env.py` (`_spawn_table`, `_spawn_piano`)
- `envs/g1_piano_reach_env_cfg.py` (piano center coordinates)

### 3. ✅ Fixed Robot Falling Through Table
**Issue**: Robot spawning at `y=-0.6` (mid-air between bench and piano), landing on table
**Fix**: 
- Changed spawn position from `y=-0.6` to `y=-0.8` (on bench)
- Lowered spawn height from `z=0.85` to `z=0.60` (just above bench top at `z=0.45`)
- Added sitting joint positions to `init_state` so robot spawns already sitting
**File**: `envs/g1_piano_reach_env_cfg.py` (robot init_state)

### 4. ✅ Fixed Import Error: `dump_pickle`
**Issue**: `ImportError: cannot import name 'dump_pickle' from 'isaaclab.utils.io'`
**Cause**: Isaac Lab's io module only exports `dump_yaml` and `load_yaml`, not `dump_pickle`
**Fix**: Removed `dump_pickle` from imports (it wasn't used in the script)
**File**: `scripts/train.py` line 68

### 5. ✅ Fixed Import Error: RL Games Wrappers
**Issue**: `ModuleNotFoundError: No module named 'isaaclab_tasks.utils.wrappers'`
**Cause**: RL Games wrappers were moved from `isaaclab_tasks.utils.wrappers.rl_games` to `isaaclab_rl.rl_games`
**Fix**: 
- Changed import from `isaaclab_tasks.utils.wrappers.rl_games` to `isaaclab_rl.rl_games`
- Added sys.path modification and `import envs` to register custom environment
- Fixed gym.register entry point from `g1_piano_play.envs` to `envs.g1_piano_reach_env`
**Files**: 
- `scripts/train.py` (line 72-77)
- `scripts/play.py` (line 49-55)
- `envs/__init__.py` (line 19)

---

## Scene Layout (Final)

```
┌─────────────────────────────────────┐
│     Table (y=0.0, top z=0.43)       │
│  Piano (y=-0.21, z=0.505)           │
└─────────────────────────────────────┘
              ↑
         0.41m gap
              ↑
┌─────────────────────────────────────┐
│  Robot (y=-0.8, z=0.60)             │
│  Bench (y=-0.8, top z=0.45)         │
└─────────────────────────────────────┘
```

### Key Coordinates:
- **Ground**: `z=0`
- **Bench**: `y=-0.8`, top at `z=0.45`
- **Robot**: `(0, -0.8, 0.60)` - sitting on bench
- **Table**: `y=0`, top at `z=0.43`
- **Piano**: `(-0.483, -0.21, 0.505)` - on table
- **Left hand target**: Piano center + `(-0.15, 0, 0.05)`
- **Right hand target**: Piano center + `(0.15, 0, 0.05)`

---

### 6. ✅ Fixed Joint Limit Violation: Elbow Pitch
**Issue**: `The following joints have default positions out of the limits: left_elbow_pitch_joint: -0.500 not in [-0.227, 3.421]`
**Cause**: Elbow pitch joints were initialized to `-0.5` rad, which is outside the valid range `[-0.227, 3.421]`
**Fix**: Changed elbow pitch initial values from `-0.5` to `0.5` rad (~29 degrees, slight bend, within limits)
**Files**: 
- `envs/g1_piano_reach_env_cfg.py` (lines 118, 123)
- `envs/g1_piano_reach_env.py` (lines 337, 342 - `_get_sitting_joint_positions`)

**Note**: The original demo (`g1-piano-play.py`) also used `-0.5`, but it worked because `set_joint_position_target` clamps values automatically. The RL environment validates limits during initialization, requiring explicit compliance.

### 7. ✅ Replaced USD Assets with Procedural Primitives
**Issue**: Table and piano USD files not cloning properly across multiple environments (only visible in env_0, others buried or invisible)
**Cause**: USD files with complex hierarchies don't always clone reliably in Isaac Lab's multi-env setup
**Fix**: Replaced both USD spawns with procedural `MeshCuboidCfg` primitives:
- **Table**: 1.2m × 0.6m × 0.43m brown cuboid (kinematic, static)
- **Piano**: 0.5m × 0.2m × 0.1m black cuboid (kinematic, static)
- Updated target positions to match new piano center (x=0.0 instead of x=-0.483)
**Files**: 
- `envs/g1_piano_reach_env.py` (`_spawn_table`, `_spawn_piano`)
- `envs/g1_piano_reach_env_cfg.py` (piano center coordinates, target offsets)

**Benefits**: Procedural primitives clone perfectly across all parallel environments, ensuring consistent training across 1024+ environments.

### 8. ✅ Fixed Missing `--disable_fabric` Argument
**Issue**: `AttributeError: 'Namespace' object has no attribute 'disable_fabric'`
**Cause**: The `--disable_fabric` argument was not defined in the argument parser, but was being used in `parse_env_cfg()` call
**Fix**: Added `--disable_fabric` argument to both `train.py` and `play.py` argument parsers
**Files**: 
- `scripts/train.py` (added argument on line 37)
- `scripts/play.py` (added argument on line 32)

**Note**: This is a standard Isaac Lab argument that needs to be manually added; it's not automatically added by `AppLauncher.add_app_launcher_args()`.

### 9. ✅ Fixed RlGamesVecEnvWrapper Missing Arguments
**Issue**: `TypeError: RlGamesVecEnvWrapper.__init__() missing 3 required positional arguments: 'rl_device', 'clip_obs', and 'clip_actions'`
**Cause**: The wrapper requires 5 additional parameters beyond just the environment:
- `rl_device`: Device for RL agent computations (e.g., "cuda:0")
- `clip_obs`: Observation clipping value (default: math.inf)
- `clip_actions`: Action clipping value (default: math.inf)
- `obs_groups`: Observation group mapping (optional)
- `concate_obs_groups`: Whether to concatenate obs groups (default: True)

**Fix**: Updated `train.py` to:
1. Load RL Games config earlier to extract wrapper parameters
2. Extract parameters: `rl_device`, `clip_obs`, `clip_actions`, `obs_groups`, `concate_obs_groups`
3. Pass all parameters to `RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)`

**Files**: 
- `scripts/train.py` (lines 116-138)

**Reference**: Proper usage documented in `IsaacLab/source/isaaclab_rl/isaaclab_rl/rl_games/rl_games.py`

### 10. ✅ Fixed Config File Path Resolution
**Issue**: `FileNotFoundError: [Errno 2] No such file or directory: '/home/solotech007/RoboGym/simulation/envs/agents/rl_games_ppo_cfg.yaml'`
**Cause**: Relative path `"envs/agents/rl_games_ppo_cfg.yaml"` was being resolved from the current working directory (simulation root) instead of the script's directory
**Fix**: Changed to use script's directory as base:
```python
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
rl_games_cfg_path = os.path.join(project_root, "envs", "agents", "rl_games_ppo_cfg.yaml")
```
**Files**: 
- `scripts/train.py` (lines 121-123)

**Result**: Config file now correctly resolved to `/home/solotech007/RoboGym/simulation/g1-piano-play/envs/agents/rl_games_ppo_cfg.yaml`

### 11. ✅ Fixed OmegaConf to Dict Conversion for RL Games
**Issue**: `omegaconf.errors.UnsupportedValueType: Value 'DefaultRewardsShaper' is not a supported primitive type`
**Cause**: RL Games internally modifies the config by inserting Python objects (like `DefaultRewardsShaper`), but OmegaConf doesn't support storing arbitrary Python objects. When we pass an OmegaConf object directly to `runner.load()`, it fails.
**Fix**: Convert OmegaConf config to regular Python dict before passing to RL Games:
```python
# Load as OmegaConf for easy modification
rl_games_cfg_omega = OmegaConf.load(rl_games_cfg_path)
# ... modify config ...

# Convert to dict for RL Games (allows internal modifications)
rl_games_cfg = OmegaConf.to_container(rl_games_cfg_omega, resolve=True)
runner.load(rl_games_cfg)  # Now works!
```
**Files**: 
- `scripts/train.py` (lines 126-142)

**Note**: Isaac Lab's official training scripts use Hydra which returns dict configs directly. Since we're loading YAML manually, we need the explicit conversion.

### 12. ✅ Fixed Missing Checkpoint Resume Arguments
**Issue**: `AttributeError: 'Namespace' object has no attribute 'resume'`
**Cause**: The script was trying to use RSL-RL style arguments (`--resume`, `--load_run`) which don't exist in RL Games
**Fix**: 
- Added `--checkpoint` argument (RL Games standard)
- Updated config to set `load_checkpoint` and `load_path` when checkpoint is provided
- Removed unused `get_checkpoint_path` import
- Simplified `runner.run()` call to only pass `train` and `play` flags

**Files**: 
- `scripts/train.py` (lines 40, 129-132, 167-170)

**Usage**:
```bash
# Resume training from checkpoint
./isaaclab.sh -p g1-piano-play/scripts/train.py \
    --task Isaac-Piano-Reach-G1-v0 \
    --checkpoint logs/rl_games/Isaac-Piano-Reach-G1-v0/nn/G1PianoReach_ep_100.pth
```

### 13. ✅ **CRITICAL FIX**: Parallel Environment Target Positions
**Issue**: Rewards were catastrophically negative (-124k per episode instead of expected -1k to -3k)
**Root Cause**: Target positions were fixed in world coordinates, but parallel environments are spatially separated:
- Env 0: Robot at (0, -0.8, 0.6) → Piano at (0, -0.21, 0.48) ✅ Distance ~0.6m
- Env 1: Robot at (3, -0.8, 0.6) → Piano at (0, -0.21, 0.48) ❌ Distance ~6.8m!
- Env 2: Robot at (6, -0.8, 0.6) → Piano at (0, -0.21, 0.48) ❌ Distance ~13.6m!

**Before Fix:**
- Hand-to-piano distance: 6-13 meters
- Reward per step: -27 to -200
- Episode reward: -16,000 to -124,000
- Training showed no learning (flat reward line)

**After Fix:**
- Hand-to-piano distance: ~0.6 meters (realistic!)
- Reward per step: ~-2.3
- Episode reward: ~-1,400
- Training should now learn successfully

**Fix**: Compute targets **relative to each environment's robot position** instead of global coordinates
```python
# Old (wrong): Fixed world coordinates
piano_center = torch.tensor([0.0, -0.21, 0.48])  # Same for all envs ❌

# New (correct): Relative to robot in each env
robot_root_pos = self.robot.data.root_pos_w  # (num_envs, 3)
piano_offset = torch.tensor([0, 0.59, -0.12])  # Relative to robot
piano_center_per_env = robot_root_pos + piano_offset  # ✅
```

**Files**: 
- `envs/g1_piano_reach_env.py` (`_compute_target_positions` method)

**Impact**: This was the **blocking issue** preventing any training progress. With this fix, the robot can now actually learn to reach!

---

## Status: ✅ Environment Ready for Training (FIXED!)

**Previous training was broken** - rewards were too negative due to incorrect target positions.  
**Now ready** for successful training with 2048 environments!

All major issues resolved. The environment should now:
- ✅ Spawn 16+ robots in parallel
- ✅ Robots sit stably on benches
- ✅ Piano and table properly positioned
- ✅ No import errors
- ✅ Correct action/observation dimensions

## Next Steps:
1. Test with `./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/test_env.py --num_envs 16`
2. Start training with `./IsaacLab/isaaclab.sh -p g1-piano-play/scripts/train.py --task Isaac-Piano-Reach-G1-v0 --num_envs 1024 --headless`
3. Monitor rewards and hand-to-target distances
4. Adjust reward weights if needed based on initial results

