# G1 Piano Playing Animation - Implementation Plan

## Executive Summary

**Objective**: Create an animation where the G1 robot performs realistic piano-playing motions:
- Hands start at opposite ends of the piano keyboard
- Perform up/down pressing motions
- Gradually move hands inward toward the center
- Pause and continue in a natural rhythm

**Feasibility**: ✅ **HIGHLY FEASIBLE** - Isaac Lab has all necessary tools

---

## Part 1: Feasibility Analysis

### ✅ Required Capabilities (All Available in Isaac Lab)

1. **Inverse Kinematics (IK)**
   - ✅ `DifferentialIKController` - Built-in differential IK solver
   - ✅ `PinkIKController` - Alternative IK solver with more features
   - ✅ Jacobian computation - Available via `Articulation.jacobian_b`
   - ✅ Multiple IK methods: pinv, svd, dls (damped least squares), transpose

2. **Object Dimension Extraction**
   - ✅ `UsdGeom.BBoxCache` - Get bounding boxes from USD prims
   - ✅ `trimesh` library - Alternative mesh analysis
   - ✅ `prim_utils.resolve_prim_pose()` - Get object positions

3. **Robot Control**
   - ✅ `Articulation.set_joint_position_target()` - Direct joint control
   - ✅ `Articulation.data.body_pose_w` - End effector positions
   - ✅ Frame management for left/right hands

### 🎯 G1 Robot Capabilities

**Arm DOF** (per arm):
- Shoulder Pitch, Roll, Yaw (3 DOF)
- Elbow Pitch, Roll (2 DOF)
- **Total: 5 DOF per arm** ✅ Sufficient for piano playing

**Workspace Limitations**:
- Estimated reach: ~0.5-0.6m from shoulder
- Good for piano keyboard (~0.3-0.5m wide playable area)

---

## Part 2: Technical Approach

### Step 1: Piano Dimension Extraction

```python
def get_piano_keyboard_bounds(stage):
    """
    Extract piano keyboard dimensions and position.
    
    Returns:
        - center_pos: (x, y, z) center of keyboard
        - width: Total width of keyboard
        - left_edge: (x, y, z) leftmost key position
        - right_edge: (x, y, z) rightmost key position
        - key_height: Height where keys should be pressed
    """
    piano_prim = stage.GetPrimAtPath("/World/Objects/Piano")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'])
    bbox = bbox_cache.ComputeWorldBound(piano_prim)
    bounds = bbox.ComputeAlignedRange()
    
    # Extract dimensions
    min_pt = bounds.GetMin()
    max_pt = bounds.GetMax()
    
    # Piano-specific offsets (keyboard is typically at front)
    # Will need to adjust based on actual piano model
    center_x = (min_pt[0] + max_pt[0]) / 2
    center_y = (min_pt[1] + max_pt[1]) / 2
    center_z = max_pt[2]  # Top of piano
    
    width = max_pt[0] - min_pt[0]  # Assuming X is keyboard width
    
    return center_pos, width, left_edge, right_edge, key_height
```

### Step 2: Inverse Kinematics Setup

**Option A: Differential IK** (Recommended for smooth animation)

```python
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

# Configure IK for left arm
left_arm_ik_cfg = DifferentialIKControllerCfg(
    command_type="pose",  # Control both position and orientation
    ik_method="dls",      # Damped least squares (handles singularities well)
    ik_params={"lambda_val": 0.1},  # Damping factor
)

left_ik_controller = DifferentialIKController(
    cfg=left_arm_ik_cfg,
    num_envs=1,
    device="cuda:0"
)

# Similar for right arm
```

**Option B: Direct Joint Control** (Simpler, may be less smooth)

```python
# Use current arm reaching pose as starting point
# Compute target hand positions
# Adjust joint angles iteratively to reach targets
```

### Step 3: Hand Frame Definition

```python
# Define end effector frames for left and right hands
# These will be the reference points for IK

# Left hand - typically at left wrist or palm
LEFT_HAND_BODY = "left_hand_link"  # Or specific link name in G1
left_hand_idx = robot.find_bodies(LEFT_HAND_BODY)[0][0]

# Right hand
RIGHT_HAND_BODY = "right_hand_link"
right_hand_idx = robot.find_bodies(RIGHT_HAND_BODY)[0][0]

# Get Jacobians for these frames
left_jacobian = robot.root_physx_view.get_jacobians()[:, left_hand_idx, :, :]
right_jacobian = robot.root_physx_view.get_jacobians()[:, right_hand_idx, :, :]
```

### Step 4: Animation State Machine

```python
class PianoPlayingAnimation:
    def __init__(self, robot, piano_bounds):
        self.state = "IDLE"
        self.states = [
            "MOVE_TO_START_POS",    # Move hands to piano edges
            "PRESS_KEY",             # Press down
            "LIFT_HAND",             # Lift up
            "MOVE_INWARD",           # Move toward center
            "PAUSE",                 # Brief pause between presses
            "FINISHED"               # Both hands at center
        ]
        
        self.left_target = piano_bounds['left_edge']
        self.right_target = piano_bounds['right_edge']
        self.center = piano_bounds['center']
        
        self.press_depth = 0.02  # 2cm press depth
        self.lift_height = 0.05  # 5cm lift height
        self.step_distance = 0.05  # 5cm per key
        
    def update(self, dt):
        if self.state == "MOVE_TO_START_POS":
            # Use IK to position hands at edges
            self.move_hands_to_targets()
            
        elif self.state == "PRESS_KEY":
            # Lower hands by press_depth
            self.left_target[2] -= self.press_depth
            self.right_target[2] -= self.press_depth
            
        elif self.state == "LIFT_HAND":
            # Raise hands
            self.left_target[2] += self.press_depth + self.lift_height
            self.right_target[2] += self.press_depth + self.lift_height
            
        elif self.state == "MOVE_INWARD":
            # Move both hands toward center
            self.left_target[0] += self.step_distance
            self.right_target[0] -= self.step_distance
            
            if distance_to_center < threshold:
                self.state = "FINISHED"
```

---

## Part 3: Incremental Implementation Steps

### Phase 1: Foundation (Days 1-2)
**Goal**: Get piano dimensions and visualize target positions

1. ✅ Create helper script to extract piano bounds
2. ✅ Visualize keyboard edges with debug spheres
3. ✅ Verify piano coordinate system

**Deliverable**: Script that prints piano dimensions and spawns markers at key positions

### Phase 2: IK Setup (Days 3-4)
**Goal**: Control one hand with IK

4. ✅ Identify G1 hand link names
5. ✅ Set up DifferentialIKController for left arm
6. ✅ Move left hand to fixed target position above piano
7. ✅ Verify hand reaches target smoothly

**Deliverable**: Script that moves left hand to specified 3D position

### Phase 3: Dual-Arm Control (Days 5-6)
**Goal**: Control both hands independently

8. ✅ Set up IK for right arm
9. ✅ Position left hand at left edge, right hand at right edge
10. ✅ Test simultaneous movement of both hands

**Deliverable**: Both hands positioned at piano edges

### Phase 4: Simple Animation (Days 7-8)
**Goal**: Basic up/down motion

11. ✅ Implement single press-lift cycle for one hand
12. ✅ Add timing and smooth interpolation
13. ✅ Extend to both hands simultaneously

**Deliverable**: Hands press down and lift up once

### Phase 5: Full Animation (Days 9-10)
**Goal**: Complete piano-playing sequence

14. ✅ Implement inward movement
15. ✅ Add pause states between actions
16. ✅ Create alternating hand pattern
17. ✅ Loop until hands meet at center

**Deliverable**: Complete piano-playing animation

### Phase 6: Polish (Days 11-12)
**Goal**: Make it look realistic

18. ✅ Tune timing parameters
19. ✅ Add hand orientation (keep fingers pointing down)
20. ✅ Smooth transitions between states
21. ✅ Add variation (different press patterns)

**Deliverable**: Polished, realistic piano-playing motion

---

## Part 4: Potential Challenges & Solutions

### Challenge 1: IK Convergence
**Problem**: IK may not converge for all target positions
**Solution**: 
- Use DLS (damped least squares) method
- Increase iteration count
- Add position limits
- Fall back to closest reachable position

### Challenge 2: Self-Collision
**Problem**: Arms may collide with torso or each other
**Solution**:
- Enable self-collision checking in G1 config
- Add workspace constraints
- Keep hands at safe distance

### Challenge 3: Piano Coordinate System
**Problem**: Piano orientation may not match expected
**Solution**:
- Add visualization/debug markers
- Allow manual offset adjustment
- Use piano front face as reference

### Challenge 4: Jacobian Computation
**Problem**: Need Jacobian for specific hand links
**Solution**:
- Use `robot.root_physx_view.get_jacobians()`
- Identify correct body indices
- Transform Jacobian to world frame if needed

### Challenge 5: Smooth Transitions
**Problem**: Jerky motion between states
**Solution**:
- Use interpolation (linear or cubic)
- Smaller step sizes
- Continuous IK updates

---

## Part 5: Alternative Approaches

### Option A: Full Differential IK (Recommended)
**Pros**: Smooth, natural motion; handles constraints
**Cons**: More complex setup; requires Jacobian computation

### Option B: Trajectory Following
**Pros**: Predictable motion; easy to debug
**Cons**: Less adaptive; may look robotic

### Option C: Hybrid Approach (Best for Learning)
**Pros**: Simple start, can add IK later
**Cons**: More work to transition

**Recommendation**: Start with Option C
1. Use direct joint control for initial positioning
2. Add simple trajectory for up/down motion
3. Upgrade to IK for more complex movements

---

## Part 6: Code Structure

```
g1-piano-play/
├── g1-piano-play.py              # Main script (existing)
├── piano_animation.py             # New: Animation controller
├── piano_ik_controller.py         # New: IK wrapper for G1 arms
├── piano_utils.py                 # New: Helper functions
└── PIANO_PLAYING_PLAN.md         # This file
```

---

## Part 7: First Implementation (Simple Version)

**Before diving into IK, let's start with a simpler approach:**

### Simple Trajectory-Based Animation

```python
def create_piano_playing_trajectory():
    """
    Create pre-defined waypoints for piano playing.
    No IK needed - just interpolate joint positions.
    """
    
    # Define key poses as joint positions
    poses = {
        'edges': {  # Hands at piano edges
            'left_arm': [...],  # Joint angles
            'right_arm': [...]
        },
        'pressed': {  # Keys pressed down
            'left_arm': [...],
            'right_arm': [...]
        },
        'lifted': {  # Hands lifted
            'left_arm': [...],
            'right_arm': [...]
        },
        'center': {  # Hands at center
            'left_arm': [...],
            'right_arm': [...]
        }
    }
    
    return poses
```

**Advantages**:
- No IK complexity
- Easy to tune manually
- Fast to implement
- Good for prototyping

---

## Recommendation: Start Simple, Add Complexity

1. **Week 1**: Implement trajectory-based animation
   - Manually define 4-5 key poses
   - Interpolate between them
   - Get basic animation working

2. **Week 2**: Add IK if needed
   - Only if manual poses are too limiting
   - Start with one arm
   - Expand to both arms

3. **Week 3**: Polish and refine
   - Tune timing
   - Add variations
   - Make it look natural

---

## Next Steps

1. **Immediate**: Create `get_piano_dimensions.py` helper script
2. **Next**: Define initial hand poses manually (no IK yet)
3. **Then**: Implement simple state machine for animation
4. **Finally**: Add IK if trajectory approach is limiting

Would you like me to start with Step 1 (piano dimensions) or proceed directly to the simple trajectory-based approach?

