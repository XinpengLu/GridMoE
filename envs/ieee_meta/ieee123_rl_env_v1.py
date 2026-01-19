import os
import sys
import random
import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding
from copy import deepcopy
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from envs.ieee_meta.ieee123_meta_env_v1 import IEEE123_Meta


class IEEE123_RL(IEEE123_Meta):
    def __init__(self, N_topo=5, N_scenerio=4, is_train=True, horizon=4, **args):
        super().__init__(N_topo=N_topo, N_scenerio=N_scenerio, **args)
        self.is_train = is_train
        self.horizon = horizon
        state, info = super().reset(1)
        state = self.preprocess_obs(state)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=state.shape)

    def preprocess_obs(self, obs, add_edge=False, print_shape=False):
        tmp = []
        tmp.append(obs['rt']['node_real_time'].reshape(-1))
        tmp.append(obs['rt']['price_real_time'])
        tmp.append(np.array([obs['rt']['time']], dtype=float))
        for e in obs['pre']['node']:
            tmp.append(e.reshape(-1))
        tmp.extend(obs['pre']['price'])

        if add_edge:
            tmp.append(np.array(obs['rt']['edge'].reshape(-1), dtype=float))
        if print_shape:
            for e in tmp:
                print(e.shape)
        return np.concatenate(tmp)

    def preprocess_info(self, state, info):
        info['cur_topo'] = np.array(state['rt']['edge'][0], dtype=int)
        return info

    def reset(self, task_num):
        prestate, info = super().reset(task_num)
        state = self.preprocess_obs(prestate)
        info = self.preprocess_info(prestate, info)
        return state, info

    def step(self, action, global_step):
        state, reward, done, tu, info = super().step(action, global_step)
        state = self.preprocess_obs(state)
        return state, reward, done, tu, info

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        self._np_random, self._np_random_seed = seeding.np_random(seed)
    

class IEEE123_RL_Vec:
    def __init__(self, num_envs=1, N_topo=5, N_scenerio=4, is_train=True, horizon=4, **args):
        self.envs = [IEEE123_RL(N_topo, N_scenerio, is_train, horizon) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        obs_shape = tuple([self.num_envs] + list(self.single_observation_space.shape))
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=obs_shape)
        act_shape = tuple([self.num_envs] + list(self.single_action_space.shape))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=act_shape)

    def reset(self, task_nums):
        assert len(task_nums) == self.num_envs
        states = []
        infos = {}
        self.task_nums = deepcopy(task_nums)
        infos['task_nums'] = self.task_nums
        infos['env_info'] = []
        for i in range(self.num_envs):
            state, info = self.envs[i].reset(task_nums[i])
            states.append(state)
            infos['env_info'].append(info)
        return np.array(states, dtype=float), infos
    
    def step(self, actions, global_step):
        assert len(actions) == self.num_envs
        states = []
        rewards = []
        dones = []
        tus = []
        infos = {}
        infos['task_nums'] = self.task_nums
        infos['env_info'] = []
        for i in range(self.num_envs):
            state, reward, done, tu, info = self.envs[i].step(actions[i], global_step)
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            tus.append(tu)
            infos['env_info'].append(info)
        
        states = np.array(states, dtype=float)
        rewards = np.array(rewards, dtype=float)
        dones = np.array(dones, dtype=bool)
        tus = np.array(tus, dtype=bool)

        return states, rewards, dones, tus, infos
    
    def close(self):
        for env in self.envs:
            env.close()
