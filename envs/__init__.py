# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 Piano Reaching RL Environment."""

import gymnasium as gym

from .g1_piano_reach_env import G1PianoReachEnv
from .g1_piano_reach_env_cfg import G1PianoReachEnvCfg

##
# Register Gym environments
##

gym.register(
    id="Isaac-Piano-Reach-G1-v0",
    entry_point="envs.g1_piano_reach_env:G1PianoReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1PianoReachEnvCfg,
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

