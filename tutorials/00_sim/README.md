# Running Isaac Lab Tutorials

## Prerequisites

The `pxr` module (USD Python bindings) is included with Isaac Sim and doesn't need separate installation.

## How to Run

Use Isaac Lab's launcher script which sets up the environment correctly:

```bash
cd /home/solotech007/RoboGym/simulation/IsaacLab
./isaaclab.sh -p /home/solotech007/RoboGym/simulation/g1-piano-play/tutorials/00_sim/spawn_prims.py
```

## What the Script Does

- Spawns a wooden table on the ground
- Places a piano on top of the table
- Creates a bench in front of the table  
- Spawns a Unitree G1 humanoid robot sitting on the bench
- Configures the robot's joints into a sitting pose facing the piano

## Troubleshooting

If you get import errors for `pxr`:
- Make sure you're using `./isaaclab.sh -p` to run the script
- Do NOT try to install `usd-core` or `pxr` via pip as it will conflict with Isaac Sim's bundled version
- The Isaac Lab launcher automatically provides access to all required USD libraries

