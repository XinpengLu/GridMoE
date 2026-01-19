import os
import sys
import copy
import csv
import warnings
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import pandapower as pp

# adjust project root if needed
proj_root = Path(__file__).resolve().parents[1]
sys.path.append(str(proj_root))

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from envs.ieee_meta.ieee123_rl_env_v1 import IEEE123_RL_Vec
from single_task import Actor, SoftQNetwork

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# ============== CONFIG = =============
# Base directory where single-task models are stored (project root relative)
BASE_DIR = Path(__file__).resolve().parent
# single_task/single_task under project root
SINGLE_TASK_MODELS_ROOT = (BASE_DIR.parent / "single_task" / "single_task").resolve()
print("SINGLE_TASK_MODELS_ROOT:", SINGLE_TASK_MODELS_ROOT)
SEEDS = [0, 1, 2, 3, 4]
# Scenario index to test (0..3). TEST_TASKS will be [scenario + 4*i for i in range(5)].
TEST_SCENARIO = [0, 1, 2, 3]
NUM_TOPO = 5
NUM_SCEN = 4
EPISODE_LEN = 96
HORIZON = 4
SWITCH_STEP = 48
DEVICE = torch.device("cpu")
OUTPUT_DIR_BASE = "ensemble_results"

def build_model_and_load(model_path, env):
    actor = Actor(env).to(DEVICE)
    qf1 = SoftQNetwork(env).to(DEVICE)
    qf2 = SoftQNetwork(env).to(DEVICE)
    qf1_target = SoftQNetwork(env).to(DEVICE)
    qf2_target = SoftQNetwork(env).to(DEVICE)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    model = torch.nn.ModuleList([qf1, qf2, qf1_target, qf2_target, actor]).to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        sd = checkpoint['state_dict']
        try:
            model[0].load_state_dict({k.replace('qf1.', ''): v for k, v in sd.items() if k.startswith('qf1.')})
            model[1].load_state_dict({k.replace('qf2.', ''): v for k, v in sd.items() if k.startswith('qf2.')})
            model[4].load_state_dict({k.replace('actor.', ''): v for k, v in sd.items() if k.startswith('actor.')})
        except Exception:
            model.load_state_dict(sd)
    else:
        model.load_state_dict(checkpoint)

    actor = model[-1]
    actor.to(DEVICE)
    actor.eval()
    return actor, model

# ------------------- 工具函数 -------------------
def normalize_line_key(k):
    parts = [p.strip() for p in k.split('-')]
    if len(parts) != 2:
        return k.strip()
    a, b = parts
    try:
        ai, bi = int(a), int(b)
        x, y = (a, b) if ai <= bi else (b, a)
    except Exception:
        x, y = (a, b) if a <= b else (b, a)
    return f"{x}-{y}"


def build_and_save_line_tables(raw_lines_per_step, switch_step=SWITCH_STEP, output_dir=None):
    T = len(raw_lines_per_step)
    if T == 0:
        return
    norm_per_step = []
    for step_dict in raw_lines_per_step:
        norm_d = {}
        for k, v in (step_dict.items() if isinstance(step_dict, dict) else []):
            k_norm = normalize_line_key(k)
            try:
                val = float(v)
            except Exception:
                val = 0.0
            norm_d[k_norm] = val
        norm_per_step.append(norm_d)

    topo1_range = range(0, min(switch_step, T))
    topo2_range = range(min(switch_step, T), T)

    keys_topo1 = set()
    for t in topo1_range:
        keys_topo1.update(norm_per_step[t].keys())
    keys_topo2 = set()
    for t in topo2_range:
        keys_topo2.update(norm_per_step[t].keys())

    def sort_key(k):
        return [int(x) for x in k.split('-')]

    keys_topo1_sorted = sorted(list(keys_topo1), key=sort_key)
    keys_topo2_sorted = sorted(list(keys_topo2), key=sort_key)

    topo1_matrix = np.zeros((len(topo1_range), len(keys_topo1_sorted)))
    for i, t in enumerate(topo1_range):
        d = norm_per_step[t]
        for j, k in enumerate(keys_topo1_sorted):
            topo1_matrix[i, j] = float(d.get(k, 0.0))

    topo2_matrix = np.zeros((len(topo2_range), len(keys_topo2_sorted)))
    for i, t in enumerate(topo2_range):
        d = norm_per_step[t]
        for j, k in enumerate(keys_topo2_sorted):
            topo2_matrix[i, j] = float(d.get(k, 0.0))

    # 保存 CSV
    df_topo1 = pd.DataFrame(topo1_matrix, columns=keys_topo1_sorted)
    df_topo2 = pd.DataFrame(topo2_matrix, columns=keys_topo2_sorted)
    df_topo1.fillna(0.0).to_csv(os.path.join(output_dir, "topo1_line_history.csv"), index_label="step")
    df_topo2.fillna(0.0).to_csv(os.path.join(output_dir, "topo2_line_history.csv"), index_label="step")

    all_keys = sorted(list(keys_topo1.union(keys_topo2)), key=sort_key)
    big_matrix = np.zeros((T, len(all_keys)))
    for j, k in enumerate(all_keys):
        if k in keys_topo1_sorted:
            idx = keys_topo1_sorted.index(k)
            big_matrix[:len(topo1_range), j] = topo1_matrix[:, idx]
        if k in keys_topo2_sorted:
            idx = keys_topo2_sorted.index(k)
            big_matrix[len(topo1_range):, j] = topo2_matrix[:, idx]

    df_big = pd.DataFrame(big_matrix, columns=[f"Line {k}" for k in all_keys])
    df_big.fillna(0.0).to_csv(os.path.join(output_dir, "line_history.csv"), index_label="step")


def load_ensemble(env, model_dir, model_indices):
    actors = []
    loaded_paths = []
    for i in model_indices:
        candidate = model_dir / f"model_{i}.pt"
        if candidate.exists():
            actor, _ = build_model_and_load(candidate, env)
            actors.append(actor)
            loaded_paths.append(candidate)
    return actors, loaded_paths


def find_model_for_task_seed(task_idx: int, seed: int):
    task_seed_dir = SINGLE_TASK_MODELS_ROOT / f"task-{task_idx}" / f"seed-{seed}"
    model = task_seed_dir / "model" / f"model_{seed}.pt"
    return model


def load_ensemble_for_seed(env, seed: int, model_task_indices: list):
    actors = []
    loaded_paths = []
    for task_idx in model_task_indices:
        mp = find_model_for_task_seed(task_idx, seed)
        actor, _ = build_model_and_load(mp, env)
        actors.append(actor)
        loaded_paths.append(mp)
    return actors, loaded_paths


def actor_get_action_numpy(actor, obs_np):
    with torch.no_grad():
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        out = actor.get_action(obs_t)
        a = out[0]
        return a.squeeze(0).cpu().numpy()

def main():
        # compute test tasks from scenario
    economic_cost_records = []
    for scenario in TEST_SCENARIO:
        TEST_TASKS = [scenario + 4 * i for i in range(5)]
        ALL_TASKS = list(range(20))
        print(f"[INFO] Testing scenario {scenario} -> test tasks: {TEST_TASKS}")
        for seed in SEEDS:
            for task in TEST_TASKS:
                print(f"\n=== Testing task {task} (seed {seed}) ===")
                env = IEEE123_RL_Vec(num_envs=1, N_topo=NUM_TOPO, N_scenerio=NUM_SCEN, is_train=False, horizon=HORIZON)
                model_task_indices = [t for t in ALL_TASKS if t not in TEST_TASKS]
                ensemble_actors, loaded_paths = load_ensemble_for_seed(env, seed, model_task_indices)
                print(f"[INFO] Loaded {len(ensemble_actors)} models for seed {seed}")

                output_dir = f"{OUTPUT_DIR_BASE}/task{task}/seed{seed}"
                os.makedirs(output_dir, exist_ok=True)
                
                obs, info = env.reset(task_nums=[task])
                gen_history = []
                sgen_history = []
                storage_history = []
                soc_history = []
                vp_history = []
                ext_history = []
                reward_history = []
                raw_line_history = []

                for step in range(EPISODE_LEN):
                    obs_tensor = torch.Tensor(obs).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        per_actions = []
                        for actor in ensemble_actors:
                            a = actor_get_action_numpy(actor, obs_tensor) 
                            per_actions.append(a)
                        per_actions = np.stack(per_actions, axis=0)
                        mean_action = np.mean(per_actions, axis=0)
                        mean_action = np.clip(mean_action, -1.0, 1.0)
                        mean_action = mean_action.squeeze(0)

                    next_obs, reward, done, tu, info = env.step(mean_action, global_step=0)
                    net = env.envs[0].current_topo.net

                    gen_history.append(net.gen['p_mw'].values.tolist())
                    sgen_history.append(net.sgen['p_mw'].values.tolist()[:10])
                    storage_history.append(net.storage['p_mw'].values.tolist())
                    soc_history.append(net.storage['soc_percent'].values.tolist())
                    vp_history.append(net.sgen['p_mw'].values.tolist()[10:])
                    ext_history.append(net.res_ext_grid['p_mw'].values)

                    res_p_line = info['env_info'][0].get('res_p_line', {})
                    raw_line_history.append(res_p_line)
                    reward_history.append(float(reward))
                    obs = next_obs
                    if done:
                        break

                economic_cost = env.envs[0].reward
                print("economic cost:", economic_cost)

                # 记录economic cost
                economic_cost_records.append({
                    'task_id': task,
                    'seed_idx': seed,
                    'economic_cost': economic_cost
                })

                # 保存发电机、负荷、储能、VP、reward CSV
                pd.DataFrame(gen_history, columns=[f"Gen {i}" for i in range(len(gen_history[0]))]) \
                    .to_csv(os.path.join(output_dir, "gen_history.csv"), index_label="step")
                pd.DataFrame(sgen_history, columns=[f"Sgen {i}" for i in range(len(sgen_history[0]))]) \
                    .to_csv(os.path.join(output_dir, "sgen_history.csv"), index_label="step")
                pd.DataFrame(storage_history, columns=[f"Storage {i}" for i in range(len(storage_history[0]))]) \
                    .to_csv(os.path.join(output_dir, "storage_history.csv"), index_label="step")
                pd.DataFrame(soc_history, columns=[f"Soc {i}" for i in range(len(soc_history[0]))]) \
                    .to_csv(os.path.join(output_dir, "soc_history.csv"), index_label="step")
                pd.DataFrame(ext_history, columns=["ext grid"]) \
                    .to_csv(os.path.join(output_dir, "ext_history.csv"), index_label="step")

                build_and_save_line_tables(raw_line_history, switch_step=SWITCH_STEP, output_dir=output_dir)
                print(f"Task{task} Seed {seed}, results saved to {output_dir}")

    # 所有任务完成后，生成统计表格
    if economic_cost_records:
        # 1. 保存所有记录的详细CSV
        detailed_df = pd.DataFrame(economic_cost_records)
        detailed_path = f"{OUTPUT_DIR_BASE}/economic_cost_detailed.csv"
        detailed_df.to_csv(detailed_path, index=False)
        print(f"\n详细记录已保存到: {detailed_path}")

        # 2. 生成统计表格（每个task五个seed的平均值、最大值、最小值）
        stats_records = []

        for task_id in ALL_TASKS:
            # 获取当前task的所有economic cost
            task_costs = [record['economic_cost'] for record in economic_cost_records
                          if record['task_id'] == task_id]

            if task_costs:
                stats_records.append({
                    'task_id': task_id,
                    'mean': sum(task_costs) / len(task_costs),
                    'max': max(task_costs),
                    'min': min(task_costs),
                    'num_seeds': len(task_costs)
                })

        # 创建统计DataFrame
        stats_df = pd.DataFrame(stats_records)

        # 保存统计表格
        stats_path = f"{OUTPUT_DIR_BASE}/economic_cost_statistics.csv"
        stats_df.to_csv(stats_path, index=False)

        print(f"统计表格已保存到: {stats_path}")
        print("\n经济成本统计结果:")
        print(stats_df.to_string(index=False))

        if len(economic_cost_records) > 0:
            all_costs = [record['economic_cost'] for record in economic_cost_records]
            overall_stats = {
                'task_id': 'Overall',
                'mean': sum(all_costs) / len(all_costs),
                'max': max(all_costs),
                'min': min(all_costs),
                'num_seeds': len(all_costs)
            }

            # 添加整体统计到表格
            overall_df = pd.DataFrame([overall_stats])
            overall_path = f"{OUTPUT_DIR_BASE}/economic_cost_overall.csv"
            overall_df.to_csv(overall_path, index=False)

            print(f"\n整体统计已保存到: {overall_path}")
            print("整体统计结果:")
            print(overall_df.to_string(index=False))

if __name__ == "__main__":
    main()
