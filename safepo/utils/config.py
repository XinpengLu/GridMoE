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


import os
import random
from distutils.util import strtobool

import numpy as np
import copy
import torch
import time
import yaml
import argparse


multi_agent_velocity_map = {
    'Safety2x4AntVelocity-v0': {
        'agent_conf': '2x4',
        'scenario': 'Ant',
    },
    'Safety4x2AntVelocity-v0': {
        'agent_conf': '4x2',
        'scenario': 'Ant',
    },
    'Safety2x3HalfCheetahVelocity-v0': {
        'agent_conf': '2x3',
        'scenario': 'HalfCheetah',
    },
    'Safety6x1HalfCheetahVelocity-v0': {
        'agent_conf': '6x1',
        'scenario': 'HalfCheetah',
    },
    'Safety3x1HopperVelocity-v0': {
        'agent_conf': '3x1',
        'scenario': 'Hopper',
    },
    'Safety2x3Walker2dVelocity-v0': {
        'agent_conf': '2x3',
        'scenario': 'Walker2d',
    },
    'Safety2x1SwimmerVelocity-v0': {
        'agent_conf': '2x1',
        'scenario': 'Swimmer',
    },
    'Safety9|8HumanoidVelocity-v0': {
        'agent_conf': '9|8',
        'scenario': 'Humanoid',
    },
}

multi_agent_goal_tasks = [
    "SafetyPointMultiGoal0-v0",
    "SafetyPointMultiGoal1-v0",
    "SafetyPointMultiGoal2-v0",
    "SafetyAntMultiGoal0-v0",
    "SafetyAntMultiGoal1-v0",
    "SafetyAntMultiGoal2-v0",
]

isaac_gym_map = {
    "ShadowHandOver_Safe_finger": "shadow_hand_over_safe_finger",
    "ShadowHandOver_Safe_joint": "shadow_hand_over_safe_joint",
    "ShadowHandCatchOver2Underarm_Safe_finger": "shadow_hand_catch_over_2_underarm_safe_finger",
    "ShadowHandCatchOver2Underarm_Safe_joint": "shadow_hand_catch_over_2_underarm_safe_joint",
    "FreightFrankaCloseDrawer": "freight_franka_close_drawer",
    "FreightFrankaPickAndPlace": "freight_franka_pick_and_place",
}

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)

def warn_task_name():
    raise Exception(
        "Unrecognized task!")

def set_seed(seed, torch_deterministic=False):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.autograd.set_detect_anomaly(True)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    return seed

def single_agent_args():
    custom_parameters = [
        {"name": "--seed", "type": int, "default":0, "help": "Random seed"},
        {"name": "--use-eval", "type": lambda x: bool(strtobool(x)), "default": False, "help": "Use evaluation environment for testing"},
        {"name": "--task", "type": str, "default": "IEEE123", "help": "The task to run"},
        {"name": "--num-envs", "type": int, "default": 1, "help": "The number of parallel game environments"},
        {"name": "--experiment", "type": str, "default": "single_agent_exp", "help": "Experiment name"},
        {"name": "--log-dir", "type": str, "default": "exps", "help": "directory to save agent logs"},
        {"name": "--device", "type": str, "default": "cuda", "help": "The device to run the model on"},
        {"name": "--device-id", "type": int, "default": 0, "help": "The device id to run the model on"},
        {"name": "--write-terminal", "type": lambda x: bool(strtobool(x)), "default": True, "help": "Toggles terminal logging"},
        {"name": "--headless", "type": lambda x: bool(strtobool(x)), "default": False, "help": "Toggles headless mode"},
        {"name": "--total-steps", "type": int, "default": 200000, "help": "Total timesteps of the experiments"},
        {"name": "--steps-per-epoch", "type": int, "default": 400, "help": "The number of steps to run in each environment per policy rollout"},
        {"name": "--randomize", "type": bool, "default": False, "help": "Wheather to randomize the environments' initial states"},
        {"name": "--cost-limit", "type": float, "default": 25.0, "help": "cost_lim"},
        {"name": "--lagrangian-multiplier-init", "type": float, "default": 0.001, "help": "initial value of lagrangian multiplier"},
        {"name": "--lagrangian-multiplier-lr", "type": float, "default": 0.035, "help": "learning rate of lagrangian multiplier"},
        {"name": "--train-tasks", "nargs": "+", "type": int, "help": "train task ids"},
        {"name": "--test-tasks", "nargs": "+", "type": int, "help": "test task ids"},
    ]
    # Create argument parser
    parser = argparse.ArgumentParser(description="RL Policy")
    issac_parameters = copy.deepcopy(custom_parameters)
    for param in custom_parameters:
        param_name = param.pop("name")
        parser.add_argument(param_name, **param)

    # Parse arguments

    args = parser.parse_args()
    cfg_env={}
    base_path = os.path.dirname(os.path.abspath(__file__)).replace("utils", "multi_agent")

    return args, cfg_env
