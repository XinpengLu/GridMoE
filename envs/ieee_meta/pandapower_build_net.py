import pandapower as pp
import numpy as np
import pandas as pd
import random
import os


class BuildNet:
    def __init__(self, opf_mode='gurobi', topo=0):
        task_set_dir = os.path.dirname(os.path.abspath(__file__))
        self.base = pd.read_csv(task_set_dir + '/task_set/base/base.csv', encoding='utf-8')
        self.bus = pd.read_csv(task_set_dir + '/task_set/base/bus_region.csv', encoding='utf-8')
        self.relax_bus = pd.read_csv(task_set_dir + '/task_set/base/relax_bus.csv', encoding='utf-8')
        self.gen = pd.read_csv(task_set_dir + '/task_set/base/gen_region.csv', encoding='utf-8')
        self.sgen = pd.read_csv(task_set_dir + '/task_set/base/sgen_region.csv', encoding='utf-8')
        self.storage = pd.read_csv(task_set_dir + '/task_set/base/storage_region.csv', encoding='utf-8')
        self.vp = pd.read_csv(task_set_dir + '/task_set/base/vp_region.csv', encoding='utf-8')

        branch_case_dir = task_set_dir + '/task_set/branch_case' + str(topo) + '.csv'
        self.branch = pd.read_csv(branch_case_dir, encoding='utf-8')
        self.branch_log_seq = [(116, 1), (13, 18), (13, 118), (18, 115), (117, 122),
                               (67, 97), (60, 119), (54, 94), (81, 86)]
        self.opf_mode = opf_mode

        self.Base_S = self.base['Base_S'][0]
        self.Base_V = self.base['Base_V'][0] / self.base['Co_V'][0]
        self.Base_R = self.Base_V ** 2 / self.Base_S

        self.time_interval = 0.25

        self.n_RL_storage = len(self.storage.loc[self.storage['Cal'] == 'RL', :])
        self.n_RL_gen = len(self.gen.loc[self.gen['Cal'] == 'RL', :])
        self.n_OPF_gen = len(self.gen.loc[self.gen['Cal'] == 'OPF', :])

        self.gen_rl_index = self.gen.loc[self.gen['Cal'] == 'RL', :].index
        self.gen_rl_max = self.gen.loc[self.gen_rl_index, 'max_P'].values
        self.gen_rl_min = self.gen.loc[self.gen_rl_index, 'min_P'].values
        self.gen_rl_cr = self.gen.loc[self.gen_rl_index, 'CR'].values

        self.gen_opf_index = self.gen.loc[self.gen['Cal'] == 'OPF', :].index
        self.gen_opf_max = self.gen.loc[self.gen_opf_index, 'max_P'].values
        self.gen_opf_min = self.gen.loc[self.gen_opf_index, 'min_P'].values
        self.gen_opf_cr = self.gen.loc[self.gen_opf_index, 'CR'].values

        self.num_load_region = []
        self.No_Seq_ren = []
        self.Type_Seq_ren = []
        self.Mode_Seq_ren = []

        for i in range(max(self.bus['Region'])):
            self.num_load_region.append(len(self.bus.loc[self.bus['Region'] == i + 1]))
        
        for index, row in self.sgen.iterrows():
            self.No_Seq_ren.append(row['Type'] + '_' + str(row['Region']) + '_' + str(row['BusNo']))
            self.Type_Seq_ren.append(row['Type'])
            self.Mode_Seq_ren.append(row['mode'])

        self.net = pp.create_empty_network(sn_mva=self.Base_S)
        self.build_net()
        self.edge_list = []
        self.Constr_edge_list = []
        self.v_list = []
        self.dg_list = []
        self.bs_list = []
        self.grid_list = []
        self.vg_list = []
        self.curtail_list = []
        self.ren_list = []
        self.relax_bus_list = []

        # params
        self.r = {}
        self.x = {}
        self.edge_Constr = {}
        self.p_DG_Max = {}
        self.p_DG_Min = {}
        self.q_DG_Max = {}
        self.q_DG_Min = {}
        self.p_DG_cr = {}
        self.p_DG_rtd = {}
        self.p_DG_rtd_rl = {}
        self.v_DG_setting = {}

        self.alpha = {}
        self.beta = {}
        self.c = {}

        self.load_beta = {}
        self.price_buy = {}

        self.load_p = {}
        self.load_q = {}

        self.vg_p_limit = {}
        self.curtail_p_limit = {}
        self.curtail_q_limit = {}

        self.ren_p = {}
        self.ren_q = {}

        self.bs_p = {}

        self.ren_seq_reg = {}
        self.load_seq_reg = {}
        self.gen_seq_reg = {}
        self.vg_seq_reg = {}
        self.cur_seq_reg = {}
        self.relax_seq_reg = {}
        self.line_seq_reg = {}
        for (m, k) in self.branch_log_seq:
            self.line_seq_reg[str(m) + '-' + str(k)] = []

    def build_net(self):
        for i in range(self.base['Sum_Bus'][0]):
            pp.create_bus(self.net, vn_kv=self.Base_V, index=int(i + 1), max_vm_pu=1.05, min_vm_pu=0.95)

        pp.create_ext_grid(self.net, bus=self.base['ext_Bus'][0], min_p_mw=self.base['min_ext_P'][0],
                           max_p_mw=self.base['max_ext_P'][0], min_q_mvar=-10, max_q_mvar=10, controllable=True)

        pp.create_poly_cost(self.net, 0, "ext_grid", cp1_eur_per_mw=60, cp0_eur=0)

        for index, row in self.branch.iterrows():
            pp.create_line_from_parameters(self.net, from_bus=int(row['from']), to_bus=int(row['to']), length_km=1,
                                           r_ohm_per_km=float(row['r_pu']) * self.Base_R,
                                           x_ohm_per_km=float(row['x_pu']) * self.Base_R, c_nf_per_km=0,
                                           max_i_ka=float(row['max_P']),
                                           in_service=True if row['state'] == True else False)

        for index, row in self.bus.iterrows():
            pp.create_load(self.net, bus=int(row['BusNo']), p_mw=float(row['Load_P_max']),
                           q_mvar=float(row['Load_Q_max']), max_p_mw=float(row['Load_P_max']),
                           max_q_mvar=float(row['Load_Q_max']), min_p_mw=0, min_q_mvar=0)

        for index, row in self.gen.iterrows():
            pp.create_gen(self.net, bus=int(row['BusNo']), p_mw=float(row['rtd_P']), max_p_mw=float(row['max_P']),
                          min_p_mw=float(row['min_P']), max_q_mvar=float(row['max_Q']), min_q_mvar=float(row['min_Q']),
                          vm_pu=float(row['V']), controllable=True if row['Cal'] == 'OPF' else False)
            pp.create_poly_cost(self.net, index, "gen", cp1_eur_per_mw=float(row['Co_1']), cp0_eur=float(row['Co_0']),
                                cp2_eur_per_mw2=float(row['Co_2']))

        for index, row in self.sgen.iterrows():
            pp.create_sgen(self.net, bus=int(row['BusNo']), p_mw=float(row['rtd_P']), q_mvar=float(row['rtd_Q']),
                           min_p_mw=float(row['min_P']), max_p_mw=float(row['max_P']), min_q_mvar=float(row['min_Q']),
                           max_q_mvar=float(row['max_Q']), controllable=True)
            pp.create_poly_cost(self.net, index, "sgen", cp1_eur_per_mw=float(row['Co_cur']), cp0_eur=0)

        for index, row in self.vp.iterrows():
            pp.create_sgen(self.net, bus=int(row['BusNo']), p_mw=float(row['rtd_cut']), q_mvar=0,
                           max_p_mw=float(row['max_Cut']), min_p_mw=0, max_q_mvar=0, min_q_mvar=0, controllable=True)
            pp.create_poly_cost(self.net, index + len(self.sgen), "sgen",
                                cp1_eur_per_mw=float(row['Co_Cut']), cp0_eur=0)

        for index, row in self.storage.iterrows():
            pp.create_storage(self.net, bus=int(row['BusNo']), p_mw=float(row['rtd_P']), q_mvar=0,
                              max_e_mwh=float(row['MaxE']), soc_percent=float(row['init_SoC']),
                              max_p_mw=float(row['max_P']), min_p_mw=float(row['min_P']),
                              max_q_mvar=float(row['max_Q']),
                              min_q_mvar=float(row['min_Q']),
                              controllable=True if row['Cal'] == 'OPF' else False)

    def update_uncontrollable_factor(self, step_dict):
        step_ren = step_dict['ren'] * self.sgen['max_P'].values
        self.net.sgen['p_mw'][0:len(self.sgen)] = step_ren
        step_ren_q = step_dict['ren'] * self.sgen['max_Q'].values
        self.net.sgen['q_mvar'][0:len(self.sgen)] = step_ren_q

        self.net.sgen['max_p_mw'][0:len(self.sgen)] = step_ren
        self.net.sgen['max_q_mvar'][0:len(self.sgen)] = step_ren_q

        step_load_p = step_dict['load'] * self.net.load['max_p_mw'].values
        self.net.load['p_mw'] = step_load_p
        step_load_q = step_dict['load'] * self.net.load['max_q_mvar'].values
        self.net.load['q_mvar'] = step_load_q

        step_price = step_dict['price']
        self.net.poly_cost.loc[self.net.poly_cost['et'] == 'ext_grid', 'cp1_eur_per_mw'] = step_price

        for index, row in self.vp.iterrows():
            self.net.sgen.loc[self.net.sgen['bus'] == int(row['BusNo']), 'max_p_mw'] = \
                self.net.load.loc[self.net.load['bus'] == int(row['BusNo']), 'p_mw'].values[0] * row['CR']

    def update_controllable_factor_rl(self, step_dict):
        action = step_dict['storage_rl']
        storage_p_rtd = np.zeros((self.n_RL_storage,))
        stor_penalty_clip_total = 0.0
        for index, row in self.storage.loc[self.storage['Cal'] == 'RL'].iterrows():
            stro_p_max = row['max_P']
            storage_p_rtd[index] = action[index] * stro_p_max
            P_attempt = action[index] * stro_p_max

            soc = self.net.storage.loc[self.net.storage['bus'] == row['BusNo'], 'soc_percent'].values[0] / 100
            dt = self.time_interval
            MaxE = row['MaxE']

            if P_attempt >= 0:
                soc_next = soc + P_attempt * row['charge_rate'] * dt / MaxE
            else:
                soc_next = soc + P_attempt * dt / MaxE / row['discharge_rate']

            P_clipped = P_attempt
            if soc_next >= row['max_SoC']:
                P_clipped = (row['max_SoC'] - soc) * MaxE / (row['charge_rate'] * dt)
                soc_next = row['max_SoC']
            elif soc_next <= row['min_SoC']:
                P_clipped = (row['min_SoC'] - soc) * MaxE * row['discharge_rate'] / dt
                soc_next = row['min_SoC']

            self.net.storage.loc[self.net.storage['bus'] == row['BusNo'], 'soc_percent'] = soc_next * 100
            self.net.storage.loc[self.net.storage['bus'] == row['BusNo'], 'p_mw'] = P_clipped

            clip_penalty = abs(P_attempt - P_clipped)
            stor_penalty_clip_total += clip_penalty

        action_gen = step_dict['gen_rl']
        gen_p_init = self.net.gen.loc[self.gen_rl_index, 'p_mw'].values
        gen_p_max = self.gen_rl_max
        gen_p_min = self.gen_rl_min
        CR = self.gen_rl_cr
        attempted_delta = action_gen * gen_p_max * CR

        allow_up = gen_p_max - gen_p_init
        allow_down = gen_p_init - gen_p_min
        violation_up = np.maximum(0.0, attempted_delta - allow_up)
        violation_down = np.maximum(0.0, -attempted_delta - allow_down)
        violation = violation_up + violation_down
        gen_violation_sum = violation.sum()

        gen_p_rtd = gen_p_init + attempted_delta

        np.clip(gen_p_rtd, gen_p_min, gen_p_max, out=gen_p_rtd)
        self.net.gen.loc[self.gen_rl_index, 'p_mw'] = gen_p_rtd

        action_ren = 0.5 - step_dict['ren_rl'] / 2
        action_cur = (1 - action_ren) * self.net.sgen['max_p_mw'][0:len(self.sgen)].values
        action_ren = action_ren * self.net.sgen['max_p_mw'][0:len(self.sgen)].values
        self.net.sgen['p_mw'][0:len(self.sgen)] = action_ren

        action_vp = step_dict['vp_rl'] / 2 + 0.5
        action_vp = action_vp * self.net.sgen['max_p_mw'][len(self.sgen):].values
        self.net.sgen['p_mw'][len(self.sgen):] = action_vp

        return gen_p_rtd, action_cur, action_vp, stor_penalty_clip_total, gen_violation_sum
