import gymnasium as gym
from gymnasium.spaces import Box
import os
import sys
import copy
import random
import math
import pandas as pd
import pandapower as pp
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from envs.ieee_meta.pandapower_build_net import BuildNet


class IEEE123_Meta(gym.Env):
    def __init__(self, N_topo=5, N_scenerio=4, is_train=True, horizon=4,
                 generator_penalty_scale=30, storage_penalty_scale=10,
                 storage_incentive_scale=10, line_penalty_scale=100, **args):
        
        self.N_topo = N_topo
        self.N_scenerio = N_scenerio
        self.is_train = is_train
        self.horizon = horizon

        self.region_num = 7
        self.region_element = {'load': [], 'ren': [], 'gen': [], 'storage': []}
        self.region_max = {'load': [], 'ren': [], 'gen': []}
        self.max_T = 96
        self.reward_vp = 0
        self.reward = 0

        # TODO
        self.gamma_ext = 0.9
        self.generator_penalty_scale = generator_penalty_scale
        self.storage_penalty_scale_list = [20, 10, 40, 2]
        self.storage_incentive_scale_list = [10, 10, 10, 2]
        self.storage_penalty_scale = storage_penalty_scale
        self.storage_incentive_scale = storage_incentive_scale
        self.line_penalty_scale = line_penalty_scale

        self.net_set = [BuildNet(opf_mode='gurobi', topo=i + 1) for i in range(self.N_topo)]

        for j in range(self.region_num):
            region_load = self.net_set[0].bus.loc[self.net_set[0].bus['Region'] == j + 1, 'BusNo'].values
            region_ren = self.net_set[0].sgen.loc[self.net_set[0].sgen['Region'] == j + 1, 'BusNo'].values
            region_gen = self.net_set[0].gen.loc[self.net_set[0].gen['Region'] == j + 1, 'BusNo'].values
            region_storage = self.net_set[0].storage.loc[self.net_set[0].storage['Region'] == j + 1, 'BusNo'].values

            ren_sum_max = np.sum(
                self.net_set[0].net.sgen.loc[self.net_set[0].net.sgen['bus'].isin(region_ren), 'max_p_mw'].values)
            load_sum_max = np.sum(
                self.net_set[0].net.load.loc[self.net_set[0].net.load['bus'].isin(region_load), 'max_p_mw'].values)
            gen_sum_max = np.sum(
                self.net_set[0].net.gen.loc[self.net_set[0].net.gen['bus'].isin(region_gen), 'max_p_mw'].values)

            self.region_element['load'].append(region_load)
            self.region_element['ren'].append(region_ren)
            self.region_element['gen'].append(region_gen)
            self.region_element['storage'].append(region_storage)

            self.region_max['load'].append(load_sum_max)
            self.region_max['ren'].append(ren_sum_max)
            self.region_max['gen'].append(gen_sum_max)

        task_set_dir = os.path.dirname(os.path.abspath(__file__))
        self.load_scenerio_task = [[pd.read_csv(
            task_set_dir + '/task_set/task' + str(scenerio_mode + 1) + '/load' + str(i + 1) + '_new.csv',
            encoding='utf-8') for i in range(5)] for scenerio_mode in range(N_scenerio)]
        self.solar_scenerio_task = [[pd.read_csv(
            task_set_dir + '/task_set/task' + str(scenerio_mode + 1) + '/solar' + str(i + 1) + '_new.csv',
            encoding='utf-8') for i in range(3)] for scenerio_mode in range(N_scenerio)]
        self.wind_scenerio_task = [[pd.read_csv(
            task_set_dir + '/task_set/task' + str(scenerio_mode + 1) + '/wind' + str(i + 1) + '_new.csv',
            encoding='utf-8') for i in range(2)] for scenerio_mode in range(N_scenerio)]
        self.price_scenerio_task = [
            pd.read_csv(task_set_dir + '/task_set/task' + str(scenerio_mode + 1) + '/price_new.csv', encoding='utf-8')
            for scenerio_mode in range(N_scenerio)]

        topo_all = pd.read_csv(task_set_dir + '/task_set/base/graph.csv', encoding='utf-8')
        self.topo_task = [topo_all.loc[:, 'node_from_topo' + str(i + 1):'node_to_topo' + str(i + 1)].values.T for i in
                          range(self.N_topo)]

        edge_all = pd.read_csv(task_set_dir + '/task_set/base/edge.csv', encoding='utf-8')
        N_all_task = self.N_scenerio * self.N_topo
        self.z2_task_state = {'node': [], 'edge_index': [], 'edge_feature': [], 'price': []}

        for i in range(N_all_task):
            z2_topo_index, z2_sce_index = self.reset_task(i)
            node_feature_100, price_feature_100 = self.make_state_task(z2_topo_index, z2_sce_index)  # [101,7,2],[101]
            topo_feature_100 = self.topo_task[z2_topo_index]
            edge_N = topo_feature_100.T
            edge_feature = []
            for e_n in range(edge_N.shape[0]):
                edge_from = edge_N[e_n, 0]
                edge_to = edge_N[e_n, 1]
                edge_feature_value = edge_all.loc[(edge_all['node_from'] == edge_from) &
                                                  (edge_all['node_to'] == edge_to), 'limit'].values[0]
                edge_feature_D = [-edge_feature_value, edge_feature_value]
                edge_feature.append(edge_feature_D)
            edge_feature_100 = np.array(edge_feature)

            self.z2_task_state['node'].append(node_feature_100)
            self.z2_task_state['edge_index'].append(topo_feature_100)
            self.z2_task_state['edge_feature'].append(edge_feature_100)
            self.z2_task_state['price'].append(price_feature_100)

        self.load_mode = {}
        self.ren_mode = {}
        self.price_mode = {}
        self.step_dict = {}
        self.state = {}

        self.current_state_rl = None
        self.cur_topo_index = None
        self.cur_sce_index = None

        self.action_space = Box(low=-1.0, high=1.0, shape=(17,))
        self.observation_space = Box(low=-1.0, high=1.0, shape=(90,))
        self.env_base = pd.read_csv(task_set_dir + '/task_set/base/baseline.csv', encoding='utf-8')

    def reset(self, task_num):
        self.task_num = task_num
        topo_index, sce_index = self.reset_task(task_num)
        self.cur_topo_index = topo_index
        self.cur_sce_index = sce_index

        self.storage_penalty_scale = self.storage_penalty_scale_list[sce_index]
        self.storage_incentive_scale = self.storage_incentive_scale_list[sce_index]
    
        self.current_topo = copy.deepcopy(self.net_set[topo_index])
        self.current_step = 0

        self.load_mode = {}
        self.ren_mode = {}
        self.price_mode = {}

        if self.is_train:
            load_random_list = [random.randint(1, 9) for _ in range(7)]
            self.load_mode['region1'] = self.load_scenerio_task[sce_index][0]['mod' + str(load_random_list[0])].values
            self.load_mode['region2'] = self.load_scenerio_task[sce_index][1]['mod' + str(load_random_list[1])].values
            self.load_mode['region3'] = self.load_scenerio_task[sce_index][2]['mod' + str(load_random_list[2])].values
            self.load_mode['region4'] = self.load_scenerio_task[sce_index][3]['mod' + str(load_random_list[3])].values
            self.load_mode['region5'] = self.load_scenerio_task[sce_index][4]['mod' + str(load_random_list[4])].values
            self.load_mode['region6'] = self.load_scenerio_task[sce_index][0]['mod' + str(load_random_list[5])].values
            self.load_mode['region7'] = self.load_scenerio_task[sce_index][1]['mod' + str(load_random_list[6])].values
            self.load_mode = pd.DataFrame(self.load_mode)

        else:
            self.load_mode['region1'] = self.load_scenerio_task[sce_index][0]['ave'].values
            self.load_mode['region2'] = self.load_scenerio_task[sce_index][1]['ave'].values
            self.load_mode['region3'] = self.load_scenerio_task[sce_index][2]['ave'].values
            self.load_mode['region4'] = self.load_scenerio_task[sce_index][3]['ave'].values
            self.load_mode['region5'] = self.load_scenerio_task[sce_index][4]['ave'].values
            self.load_mode['region6'] = self.load_scenerio_task[sce_index][0]['ave'].values
            self.load_mode['region7'] = self.load_scenerio_task[sce_index][1]['ave'].values

            self.load_mode = pd.DataFrame(self.load_mode)

        if self.is_train:
            sgen_random_list = [random.randint(1, 9) for _ in range(len(self.current_topo.No_Seq_ren))]
            for i in range(len(self.current_topo.No_Seq_ren)):
                if self.current_topo.Type_Seq_ren[i] == 'PV':
                    if self.current_topo.Mode_Seq_ren[i] == 1:
                        self.ren_mode[self.current_topo.No_Seq_ren[i]] = self.solar_scenerio_task[sce_index][0][
                            'mod' + str(sgen_random_list[i])].values
                    elif self.current_topo.Mode_Seq_ren[i] == 2:
                        self.ren_mode[self.current_topo.No_Seq_ren[i]] = self.solar_scenerio_task[sce_index][1][
                            'mod' + str(sgen_random_list[i])].values
                    elif self.current_topo.Mode_Seq_ren[i] == 3:
                        self.ren_mode[self.current_topo.No_Seq_ren[i]] = self.solar_scenerio_task[sce_index][2][
                            'mod' + str(sgen_random_list[i])].values
                if self.current_topo.Type_Seq_ren[i] == 'WT':
                    if self.current_topo.Mode_Seq_ren[i] == 1:
                        self.ren_mode[self.current_topo.No_Seq_ren[i]] = self.wind_scenerio_task[sce_index][0][
                            'mod' + str(sgen_random_list[i])].values
                    elif self.current_topo.Mode_Seq_ren[i] == 2:
                        self.ren_mode[self.current_topo.No_Seq_ren[i]] = self.wind_scenerio_task[sce_index][1][
                            'mod' + str(sgen_random_list[i])].values
            self.ren_mode = pd.DataFrame(self.ren_mode)

        else:
            for i in range(len(self.current_topo.No_Seq_ren)):
                if self.current_topo.Type_Seq_ren[i] == 'PV':
                    if self.current_topo.Mode_Seq_ren[i] == 1:
                        self.ren_mode[self.current_topo.No_Seq_ren[i]] = self.solar_scenerio_task[sce_index][0][
                            'ave'].values
                    elif self.current_topo.Mode_Seq_ren[i] == 2:
                        self.ren_mode[self.current_topo.No_Seq_ren[i]] = self.solar_scenerio_task[sce_index][1][
                            'ave'].values
                    elif self.current_topo.Mode_Seq_ren[i] == 3:
                        self.ren_mode[self.current_topo.No_Seq_ren[i]] = self.solar_scenerio_task[sce_index][2][
                            'ave'].values
                if self.current_topo.Type_Seq_ren[i] == 'WT':
                    if self.current_topo.Mode_Seq_ren[i] == 1:
                        self.ren_mode[self.current_topo.No_Seq_ren[i]] = self.wind_scenerio_task[sce_index][0][
                            'ave'].values
                    elif self.current_topo.Mode_Seq_ren[i] == 2:
                        self.ren_mode[self.current_topo.No_Seq_ren[i]] = self.wind_scenerio_task[sce_index][1][
                            'ave'].values
            self.ren_mode = pd.DataFrame(self.ren_mode)

        if self.is_train:
            price_random = random.randint(1, 9)
            self.price_mode['ext_buy_price'] = self.price_scenerio_task[sce_index]['mod' + str(price_random)].values
        else:
            self.price_mode['ext_buy_price'] = self.price_scenerio_task[sce_index]['ave'].values

        self.pre_feature = {'node': [], 'price': []}
        for t in range(100):
            a = self.load_mode.loc[t].values  # [7,]
            step_load_rate = []
            for i in range(a.shape[0]):
                step_load_rate += [a[i]] * self.current_topo.num_load_region[i]
            step_load_rate = np.array(step_load_rate)
            step_ren_rate = self.ren_mode.loc[t, self.current_topo.No_Seq_ren].values  # [10,]
            step_price_rate = self.price_mode['ext_buy_price'][t]  # [1,]

            self.step_dict['load'] = step_load_rate
            self.step_dict['ren'] = step_ren_rate
            self.step_dict['price'] = step_price_rate

            self.current_topo.update_uncontrollable_factor(step_dict=self.step_dict)

            state_pre = self.make_state_prediction(topo_index, t)
            self.pre_feature['node'].append(state_pre['node'])
            self.pre_feature['price'].append(state_pre['price'])

        a = self.load_mode.loc[self.current_step].values
        step_load_rate = []
        for i in range(a.shape[0]):
            step_load_rate += [a[i]] * self.current_topo.num_load_region[i]
        step_load_rate = np.array(step_load_rate)
        step_ren_rate = self.ren_mode.loc[self.current_step, self.current_topo.No_Seq_ren].values
        step_price_rate = self.price_mode['ext_buy_price'][self.current_step]
        self.step_dict['ren'] = step_ren_rate
        self.step_dict['load'] = step_load_rate
        self.step_dict['price'] = step_price_rate
        self.current_topo.update_uncontrollable_factor(step_dict=self.step_dict)
        state_real_time = self.make_state_rl(self.cur_topo_index)

        state_prediction_node = [self.pre_feature['node'][(self.current_step + 1 + i) % self.max_T] for i in
                                 range(self.horizon)]
        state_prediction_price = [self.pre_feature['price'][(self.current_step + 1 + i) % self.max_T] for i in
                                  range(self.horizon)]

        self.state['rt'] = state_real_time
        self.state['pre'] = {'node': state_prediction_node, 'price': state_prediction_price}
        info = {}
        return self.state, info

    def step(self, action, global_step):
        action = np.clip(action, -1.0, 1.0)
        self.step_dict['storage_rl'] = action[:3]
        self.step_dict['gen_rl'] = action[3:7]
        self.step_dict['ren_rl'] = action[7:17]
        self.step_dict['vp_rl'] = np.full(5, -1.0)

        self.current_topo.update_uncontrollable_factor(step_dict=self.step_dict)
        gen_p_rtd, p_cur, p_vp, stor_clip_sum, gen_violation_sum = self.current_topo.update_controllable_factor_rl(step_dict=self.step_dict)
        if not self.is_train:
            stor_clip_sum = 0
            gen_violation_sum = 0

        stor_penalty = -self.storage_penalty_scale * stor_clip_sum
        gen_penalty = -self.generator_penalty_scale * gen_violation_sum

        try:
            pp.runpp(self.current_topo.net, numba=True)
        except:
            reward = -100 + stor_penalty + gen_penalty
            info = {}
            done = True
            tu = True
            return self.state, reward, done, tu, info
        else:
            reward = self.reward_func(gen_p_rtd, p_cur, p_vp, global_step) + stor_penalty + gen_penalty

            if self.is_train:
                r_ext = self.compute_prediction_reward()
                reward = reward + r_ext * self.storage_incentive_scale

            def compute_line_violation(res_p_line, max_p_line):
                violations = []
                for i, (p, limit) in enumerate(zip(res_p_line, max_p_line)):
                    violations.append(max(0.0, p - limit))
                return np.array(violations)

            if self.is_train:
                res_p_line_list = []
                max_p_line_list = []
                for i, d in self.current_topo.net.line[-8:].iterrows():
                    if d['in_service']:
                        actual_p = self.current_topo.net.res_line.loc[i, 'p_from_mw']
                        max_p = self.current_topo.branch.loc[i, 'max_P']
                        res_p_line_list.append(actual_p)
                        max_p_line_list.append(max_p)

                line_violations = compute_line_violation(res_p_line_list, max_p_line_list)
                total_line_violation = line_violations.sum()
                line_penalty = -self.line_penalty_scale * total_line_violation
                reward += line_penalty

            info = {}
            info['res_V_node'] = np.array(self.current_topo.net.res_bus['vm_pu'].values).reshape(-1)
            info['V_node_max'] = 1.05
            info['V_node_min'] = 0.95
            info['res_p_line'] = {}
            info['max_p_line'] = {}
            info['res_V_outline'] = (
                np.sum(info['res_V_node'] < info['V_node_min']) +
                np.sum(info['res_V_node'] > info['V_node_max'])
            ) / info['res_V_node'].shape[0]

            info['res_p_outline'] = 0
            info['line_in_service'] = 0

            for i, d in self.current_topo.net.line[-8:].iterrows():
                if d['in_service']:
                    key = f"{d['from_bus']}-{d['to_bus']}"
                    info['res_p_line'][key] = self.current_topo.net.res_line.loc[i, 'p_from_mw']
                    info['max_p_line'][key] = self.current_topo.branch.loc[i, 'max_P']
                    info['line_in_service'] += 1
                    if info['max_p_line'][key] < info['res_p_line'][key]:
                        info['res_p_outline'] += 1

            info['res_p_outline'] /= max(1, info['line_in_service'])
            info['res_gen_p'] = np.array(self.current_topo.net.gen['p_mw'].values.tolist())
            info['res_sgen_p'] = np.array(self.current_topo.net.sgen['p_mw'].values.tolist()[:10])
            info['res_storage_p'] = np.array(self.current_topo.net.storage['p_mw'].values.tolist())
            info['res_storage_soc'] = np.array(self.current_topo.net.storage['soc_percent'].values.tolist())
            info['res_vp_p'] = np.array(self.current_topo.net.sgen['p_mw'].values.tolist()[10:])
            info['ext_grid_p'] = np.array(self.current_topo.net.res_ext_grid['p_mw'].values.tolist())


        self.current_step += 1
        if self.current_step >= 96:
            done = True
        else:
            done = False

        a = self.load_mode.loc[self.current_step].values
        step_load_rate = []
        for i in range(a.shape[0]):
            step_load_rate += [a[i]] * self.current_topo.num_load_region[i]
        step_load_rate = np.array(step_load_rate)
        step_ren_rate = self.ren_mode.loc[self.current_step, self.current_topo.No_Seq_ren].values
        step_price_rate = self.price_mode['ext_buy_price'][self.current_step]
        self.step_dict['ren'] = step_ren_rate
        self.step_dict['load'] = step_load_rate
        self.step_dict['price'] = step_price_rate
        self.current_topo.update_uncontrollable_factor(step_dict=self.step_dict)
        state_real_time = self.make_state_rl(self.cur_topo_index)

        state_prediction_node = [self.pre_feature['node'][(self.current_step + i + 1) % self.max_T] for i in
                                 range(self.horizon)]
        state_prediction_price = [self.pre_feature['price'][(self.current_step + i + 1) % self.max_T] for i in
                                  range(self.horizon)]

        self.state['rt'] = state_real_time
        self.state['pre'] = {'node': state_prediction_node, 'price': state_prediction_price}

        tu = False
        return self.state, reward, done, tu, info

    def render(self, mode='human'):
        pass

    def make_state_rl(self, topo_index):
        current_state_dict = {}
        current_graph_state = []
        current_ext_feature = np.array([self.price_mode['ext_buy_price'][self.current_step] / 80])
        current_edge_index = self.topo_task[topo_index]

        for j in range(self.region_num):
            ren_sum = np.sum(self.current_topo.net.sgen.loc[self.current_topo.net.sgen['bus'].isin(
                self.region_element['ren'][j]), 'max_p_mw'].values)
            load_sum = np.sum(self.current_topo.net.load.loc[self.current_topo.net.load['bus'].isin(
                self.region_element['load'][j]), 'p_mw'].values)
            gen_sum = np.sum(self.current_topo.net.gen.loc[
                                 self.current_topo.net.gen['bus'].isin(self.region_element['gen'][j]), 'p_mw'].values)
            storage_soc = np.sum(self.current_topo.net.storage.loc[self.current_topo.net.storage['bus'].isin(
                self.region_element['storage'][j]), 'soc_percent'].values) / 100

            node_feature = [ren_sum / max(self.region_max['ren']), load_sum / max(self.region_max['load']),
                            gen_sum / max(self.region_max['gen']), storage_soc]
            current_graph_state.append(np.array(node_feature))
        current_graph_state = np.array(current_graph_state)

        current_state_dict['node_real_time'] = current_graph_state
        current_state_dict['price_real_time'] = current_ext_feature
        current_state_dict['edge'] = current_edge_index
        current_state_dict['time'] = self.current_step

        """
        {
          'node_real_time': np.ndarray(shape=(region_num, 4)),  # per-region features
          'price_real_time': np.ndarray(shape=(1,)),           # normalized price scalar
          'edge': edge_index,                                  # topo edge index (E×2)
          'time': current_step
        }
        """

        return current_state_dict

    def make_state_prediction(self, topo_index, t):
        current_state_dict = {}
        current_graph_state = []
        current_ext_feature = np.array([self.price_mode['ext_buy_price'][t] / 80])
        current_edge_index = self.topo_task[topo_index]

        for j in range(self.region_num):
            ren_sum = np.sum(self.current_topo.net.sgen.loc[self.current_topo.net.sgen['bus'].isin(
                self.region_element['ren'][j]), 'max_p_mw'].values)
            load_sum = np.sum(self.current_topo.net.load.loc[self.current_topo.net.load['bus'].isin(
                self.region_element['load'][j]), 'p_mw'].values)

            node_feature = [ren_sum / max(self.region_max['ren']), load_sum / max(self.region_max['load'])]
            current_graph_state.append(np.array(node_feature))
        current_graph_state = np.array(current_graph_state)

        current_state_dict['node'] = current_graph_state
        current_state_dict['price'] = current_ext_feature
        """
        'node': np.ndarray(shape=(7, 2)), 'price': np.ndarray(shape=(1,)) 
        """
        return current_state_dict

    def make_state_task(self, topo_index, sce_index):
        net_task = self.net_set[topo_index]
        load_task = self.load_scenerio_task[sce_index]
        solar_task = self.solar_scenerio_task[sce_index]
        wind_task = self.wind_scenerio_task[sce_index]
        task_seq = []

        for j in range(self.region_num):
            load = self.region_max['load'][j]
            load_mode = load_task[j % 5]['ave'].values
            load_seq = load * load_mode   
            if j + 1 in net_task.sgen['Region']:
                sgen_seq = np.zeros((101,))
                for i, d in net_task.sgen.iterrows():
                    if d['Region'] == j + 1:
                        if d['Type'] == 'PV':
                            sgen = d['max_P']
                            sgen_mode = solar_task[d['mode'] - 1]['ave'].values
                            sgen_seq += sgen * sgen_mode
                        if d['Type'] == 'WT':
                            sgen = d['max_P']
                            sgen_mode = wind_task[d['mode'] - 1]['ave'].values
                            sgen_seq += sgen * sgen_mode
            else:
                sgen_seq = np.zeros((len(solar_task[0][0]),))

            region_seq = np.concatenate((load_seq[np.newaxis, :] / max(self.region_max['load']),
                                         sgen_seq[np.newaxis, :] / max(self.region_max['ren'])), axis=0).T  # [101, 2]
            task_seq.append(region_seq)

        task_seq = [item[:, np.newaxis, :] for item in task_seq]
        task_seq = np.concatenate(task_seq, axis=1)  # [101, 7, 2]

        price_seq = self.price_scenerio_task[sce_index]['ave'].values / 80  # [107,]

        return task_seq, price_seq

    def reward_func(self, gen_rtd, p_cur, p_vp, global_step):
        gen_rl = self.current_topo.gen.loc[self.current_topo.gen['Cal'] == 'RL', :]
        cost_gen = 0
        for n_gen in range(gen_rtd.shape[0]):
            cost_gen += (gen_rl['Co_2'].values[n_gen] * gen_rtd[n_gen] ** 2
                         + gen_rl['Co_1'].values[n_gen] * gen_rtd[n_gen]
                         + gen_rl['Co_0'].values[n_gen])

        cost_cur = 0
        for i, d in self.current_topo.sgen.iterrows():
            cost_cur += p_cur[i] * d['Co_cur']

        cost_vp = 0
        for i, d in self.current_topo.vp.iterrows():
            cost_vp += p_vp[i] * d['Co_Cut']
        self.reward_vp += cost_vp

        cost_pl = np.sum(abs(self.current_topo.net.res_line['pl_mw'].values)) * 20

        p_ext = self.current_topo.net.res_ext_grid['p_mw'].values[0]
        price = self.step_dict['price']
        if p_ext >= 0:
            cost_ext = price * p_ext
        else:
            cost_ext = abs(p_ext) * (100 if self.is_train else 0)

        reward = -(cost_gen + cost_cur + cost_vp + cost_pl + cost_ext)
        self.reward += reward
        return reward


    def compute_prediction_reward(self):
        storage_powers = np.array(self.current_topo.net.storage['p_mw'].values)

        discounted_price = self.compute_discounted_price_factor()
        discounted_load = self.compute_discounted_load_factor()
        discounted_ren = self.compute_discounted_ren_factor()

        F_d = (discounted_price * discounted_load) / discounted_ren

        price_base = float(np.mean(self.pre_feature['price']))
        all_loads = np.array([np.mean(nf[:, 1]) for nf in self.pre_feature['node']])
        load_base = float(np.mean(all_loads))
        all_ren = np.array([np.mean(nf[:, 0]) for nf in self.pre_feature['node']])
        ren_base = float(np.mean(all_ren))

        baseline = (price_base * load_base) / ren_base

        total_r_ext = 0.0
        for P in storage_powers:
            total_r_ext += (F_d - baseline) * (-P)
        return total_r_ext

    def compute_discounted_price_factor(self):
        weights = [self.gamma_ext ** tau for tau in range(self.horizon + 1)]
        weighted_sum = 0.0
        total_weight = sum(weights)
        for i, weight in enumerate(weights):
            t_index = (self.current_step + i) % self.max_T
            future_price = self.pre_feature['price'][t_index][0]
            weighted_sum += weight * future_price
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def compute_discounted_load_factor(self):
        weights = [self.gamma_ext ** tau for tau in range(self.horizon + 1)]
        weighted_sum = 0.0
        total_weight = sum(weights)
        for i, weight in enumerate(weights):
            t_index = (self.current_step + i) % self.max_T
            future_node_feature = self.pre_feature['node'][t_index]
            future_load = np.mean(future_node_feature[:, 1]) if len(future_node_feature.shape) > 1 else 0
            weighted_sum += weight * future_load
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def compute_discounted_ren_factor(self):
        weights = [self.gamma_ext ** tau for tau in range(self.horizon + 1)]
        weighted_sum = 0.0
        total_weight = sum(weights)
        for i, weight in enumerate(weights):
            t_index = (self.current_step + i) % self.max_T
            future_node_feature = self.pre_feature['node'][t_index]
            future_ren = np.mean(future_node_feature[:, 0]) if len(future_node_feature.shape) > 1 else 0
            weighted_sum += weight * future_ren
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def reset_task(self, n_task):
        topo_index = n_task // self.N_scenerio
        sce_index = n_task % self.N_scenerio
        return topo_index, sce_index

    def close(self):
        pass
