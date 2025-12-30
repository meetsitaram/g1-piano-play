# Tomorrow's Action Items

## Quick Start Checklist ✅

### 1. Review Documentation (15 minutes)
- [ ] Read `README.md` - Complete project overview
- [ ] Skim `tutorials/00_sim/PIANO_PLAYING_PLAN.md` - Detailed technical plan

### 2. Extract Piano Dimensions (10 minutes)
```bash
cd /home/solotech007/RoboGym/simulation
./IsaacLab/isaaclab.sh -p g1-piano-play/tutorials/00_sim/get_piano_dimensions.py
```

**What to capture**:
- Keyboard width
- Left edge X coordinate
- Right edge X coordinate
- Center X coordinate
- Key height Z coordinate
- Copy the Python dictionary output

### 3. Make a Decision (5 minutes)
Choose implementation approach:

**Option A: Trajectory-based** (Recommended)
- Faster (3-5 days)
- Easier to debug
- Good for demo/prototype

**Option B: IK-based**
- More complex (7-10 days)
- More flexible
- Better for precise control

### 4. Start Implementation

#### If Choosing Trajectory (Option A):

1. Create new function in `g1-piano-play.py`:
```python
def get_piano_edge_poses(piano_dims):
    """
    Define hand positions at piano edges.
    
    Args:
        piano_dims: Dictionary from get_piano_dimensions.py
    
    Returns:
        dict: Joint positions for hands at keyboard edges
    """
    # Start with current arm reaching pose
    edge_pose = get_arm_reaching_positions().copy()
    
    # Adjust shoulder/elbow angles to position hands at edges
    # Left hand more to the left
    edge_pose['left_shoulder_yaw_joint'] = 0.2  # Rotate outward
    edge_pose['left_shoulder_roll_joint'] = 0.3  # Move left
    
    # Right hand more to the right  
    edge_pose['right_shoulder_yaw_joint'] = -0.2  # Rotate outward
    edge_pose['right_shoulder_roll_joint'] = -0.3  # Move right
    
    return edge_pose
```

2. Test the pose:
```python
# In main(), after sitting pose
edge_positions = get_piano_edge_poses(piano_dims)
apply_joint_positions(robot, joint_targets, edge_positions, 
                      dof_names, sim, sim_dt, 
                      settle_steps=200, description="edge positions")
```

#### If Choosing IK (Option B):

1. Study example code:
```bash
# Look at these files for IK usage examples
cat IsaacLab/source/isaaclab/isaaclab/controllers/differential_ik.py
cat IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py | grep -A 20 "set_pos_inverse_kinematics"
```

2. Identify hand bodies:
```python
# Add to main() after robot creation
print("\n[DEBUG]: Available bodies:")
for i, name in enumerate(robot.body_names):
    if 'hand' in name.lower() or 'palm' in name.lower() or 'wrist' in name.lower():
        print(f"  [{i}] {name}")
```

---

## Files Created Today

1. **`README.md`** (Main documentation)
   - Complete project overview
   - Current status
   - Implementation approaches
   - Phase-by-phase plan
   - Timeline estimates
   - Troubleshooting guide

2. **`tutorials/00_sim/PIANO_PLAYING_PLAN.md`** (Technical details)
   - Feasibility analysis
   - IK implementation details
   - Code examples
   - Challenges and solutions

3. **`tutorials/00_sim/get_piano_dimensions.py`** (Helper script)
   - Extracts piano bounding box
   - Calculates keyboard dimensions
   - Suggests hand positions
   - Outputs Python dictionary

4. **`NEXT_STEPS.md`** (This file)
   - Quick action items
   - Decision points
   - Code snippets to get started

---

## Current Working Files

- **`tutorials/00_sim/g1-piano-play.py`** - Main simulation (541 lines)
  - Scene setup ✅
  - Robot sitting pose ✅
  - Arm reaching pose ✅
  - Ready for animation phase

- **`tutorials/00_sim/g1-piano-play-bkp.py`** - Backup of working version

---

## Quick Reference Commands

### Run Simulation
```bash
cd /home/solotech007/RoboGym/simulation
./IsaacLab/isaaclab.sh -p g1-piano-play/tutorials/00_sim/g1-piano-play.py
```

### Extract Piano Dimensions
```bash
./IsaacLab/isaaclab.sh -p g1-piano-play/tutorials/00_sim/get_piano_dimensions.py
```

### Backup Before Changes
```bash
cd g1-piano-play/tutorials/00_sim/
cp g1-piano-play.py g1-piano-play-backup-$(date +%Y%m%d-%H%M).py
```

---

## Key Design Decisions to Make

1. **Trajectory vs IK?**
   - Trajectory: Faster, simpler
   - IK: More flexible, complex

2. **How many key poses?**
   - Minimum: 3 (edges, pressing, lifted)
   - Recommended: 5-6 (edges, press, lift, move, pause, center)

3. **Animation style?**
   - Simultaneous (both hands together)
   - Alternating (left, right, left, right)
   - Random pattern

4. **Movement parameters?**
   - Press depth: 1-3cm
   - Lift height: 3-5cm
   - Horizontal step: 3-5cm per key
   - Press duration: 0.2-0.5s

---

## Estimated Time Breakdown

### Phase 1 (Tomorrow - Day 1)
- Extract dimensions: **30 min**
- Define edge poses: **2 hours**
- Test poses: **1 hour**
- **Total: 3.5 hours**

### Phase 2 (Day 2)
- Add press/lift poses: **2 hours**
- Implement interpolation: **2 hours**
- **Total: 4 hours**

### Phase 3 (Day 3)
- State machine: **3 hours**
- Testing & tuning: **2 hours**
- **Total: 5 hours**

### Phase 4 (Day 4-5)
- Polish & refinement: **4-6 hours**

**Total for MVP**: ~16-20 hours (spread over 4-5 days)

---

## Success Metrics

### End of Tomorrow (Day 1)
- [ ] Piano dimensions documented
- [ ] Hands positioned at keyboard edges
- [ ] No robot falling or collisions
- [ ] One hand can press/lift (even if rough)

### End of Week 1
- [ ] Both hands move from edges to center
- [ ] Up/down pressing motion works
- [ ] Smooth interpolation between poses
- [ ] Basic animation loop complete

---

## Potential Issues & Quick Fixes

### Issue: Piano dimensions don't make sense
**Fix**: Check piano orientation, may need to swap axes

### Issue: Hands can't reach edges
**Fix**: Adjust `robot_offset_from_bench` or `bench_distance_from_table`

### Issue: Robot falls forward
**Fix**: Reduce arm reach or increase `torso_joint` angle

### Issue: Jerky motion
**Fix**: Increase interpolation steps or use cubic interpolation

---

## Resources for Reference

### Code Examples
- Current sitting function: Line 109-140 in `g1-piano-play.py`
- Current arm reaching function: Line 143-167
- Joint application: Line 171-207

### Isaac Lab Controllers
- `isaaclab/controllers/differential_ik.py`
- `isaaclab/controllers/differential_ik_cfg.py`
- `isaaclab_tasks/direct/factory/factory_env.py` (line 514-551)

### G1 Configuration
- `IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py`
- Line 272-385: G1_CFG and G1_MINIMAL_CFG

---

## Final Checklist Before Starting

- [ ] Documentation reviewed
- [ ] Piano dimensions extracted
- [ ] Approach decided (Trajectory or IK)
- [ ] Backup created
- [ ] Ready to code!

---

## Questions? Check These First

1. **"How do I find joint names?"**
   - Run simulation, check console output line 504-507
   - Or add: `print(robot.data.joint_names)`

2. **"How do I visualize hand positions?"**
   - Use debug spheres (see PIANO_PLAYING_PLAN.md)
   - Or print hand positions: `print(robot.data.body_pose_w[:, hand_idx])`

3. **"Animation not smooth?"**
   - Increase `settle_steps` parameter
   - Add more interpolation steps
   - Use cubic instead of linear interpolation

4. **"IK not converging?"**
   - Use DLS method instead of pinv
   - Increase damping factor
   - Check if target is reachable

---

## Good Luck! 🚀

Everything is set up and ready. The foundation is solid, and the plan is clear. 

**Start with the simple approach** (trajectory), get it working, then enhance if needed.

**Don't overthink** - get hands moving first, polish later.

**Remember**: Perfect is the enemy of good. A working demo beats a perfect plan!

---

Last updated: December 29, 2025
Next review: Tomorrow morning

