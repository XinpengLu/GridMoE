# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations
from typing import Callable
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from envs.ieee_meta.ieee123_rl_env_v1 import IEEE123_SafeRL, IEEE123_SafeRL_Vec


def make_ieee123_saferl_env(num_envs=1, use_vec=True, cost_type='lineP', cost_coef=1.0):
    if not use_vec and num_envs == 1:
        env = IEEE123_SafeRL(cost_type=cost_type, cost_coef=cost_coef)
        obs_space = env.observation_space
        act_space = env.action_space
    else:
        env = IEEE123_SafeRL_Vec(num_envs=num_envs)
        obs_space = env.single_observation_space
        act_space = env.single_action_space

    return env, obs_space, act_space
