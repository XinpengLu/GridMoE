# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
import signal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from copy import deepcopy
from envs.ieee_meta.ieee123_rl_env_v1 import IEEE123_RL_Vec
from envs.ieee_meta.shmem_vec_env import IEEE123_RL_ShmVec
import warnings
from utils import write2json, task_nums2str
from safepo.common.logger import EpochLogger
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "norm_moe"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "IEEE123Topology"
    """the environment id of the task"""
    total_timesteps: int = 200000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    use_norm: bool = True
    """if use normalization"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 5e-3
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 200
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    test_freq: int = 2000
    save_freq: int = 10000
    save_model: bool = False
    """model save frequency"""
    log_dir: str = "norm_moe"
    train_tasks: tuple[int, ...] = (15,)
    test_tasks: tuple[int, ...] = (15,)
    add_edge: bool = False
    mix_num: int = 5
    topk: int = 2
    critic_aux_loss_weight: float = 1.0
    actor_aux_loss_weight: float = 1.0
    horizon: int = 4
    generator_penalty_scale: float = 30
    storage_penalty_scale: float = 10
    storage_incentive_scale: float = 10
    line_penalty_scale: float = 100


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, mix_num, topk, use_norm=True, hidden=256, activation=F.relu):
        super().__init__()
        obs_dim = int(np.array(env.single_observation_space.shape).prod())
        act_dim = int(np.prod(env.single_action_space.shape))
        self.hidden = hidden
        self.activation = activation
        self.fc1 = nn.Linear(obs_dim + act_dim, self.hidden * mix_num)
        self.fc2 = nn.Linear(self.hidden * mix_num, self.hidden * mix_num)
        self.fc3 = nn.Linear(self.hidden, 1)

        self.norm1 = nn.LayerNorm(self.hidden * mix_num) if use_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(self.hidden * mix_num) if use_norm else nn.Identity()

        self.gate = nn.Linear(env.single_observation_space.shape[0] + env.single_action_space.shape[0], mix_num)
        self.mix_num = mix_num
        self.topk = topk
        self.obs_len = env.single_observation_space.shape[0]
        self.expert_usage = None
        self.total_calls = 0

    def forward(self, x, a):
        batch = x.shape[0]
        if self.expert_usage is None:
            self.expert_usage = torch.zeros(self.mix_num, device=x.device)
        x = torch.cat([x, a], 1)
        gate_scores = F.softmax(self.gate(x), dim=-1)
        topk_vals, topk_idx = torch.topk(gate_scores, k=self.topk, dim=-1)
        with torch.no_grad():
            self.total_calls += batch
            for i in range(batch):
                self.expert_usage[topk_idx[i]] += 1

        x = self.activation(self.norm1(self.fc1(x)))
        x = self.activation(self.norm2(self.fc2(x)))
        x = x.reshape(batch, self.hidden, self.mix_num)
        topk_vals = topk_vals / torch.sum(topk_vals, dim=-1, keepdim=True)
        scores = torch.zeros_like(gate_scores)
        scores[torch.arange(batch).view(-1, 1), topk_idx] = topk_vals
        scores = scores.reshape(batch, 1, -1)
        x = torch.mean(x * scores, dim=-1)
        x = self.fc3(x)
        return x
    
    def get_expert_usage_stats(self):
        """获取专家使用统计信息"""
        if self.total_calls == 0:
            return torch.zeros_like(self.expert_usage)
        return self.expert_usage / self.total_calls


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, mix_num, topk, use_norm=True, hidden=256, activation=F.relu):
        super().__init__()
        obs_dim = int(np.array(env.single_observation_space.shape).prod())
        act_dim = int(np.prod(env.single_action_space.shape))
        self.hidden = hidden
        self.activation = activation

        self.fc1 = nn.Linear(obs_dim, self.hidden * mix_num)
        self.fc2 = nn.Linear(self.hidden * mix_num, self.hidden * mix_num)

        self.norm1 = nn.LayerNorm(self.hidden * mix_num) if use_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(self.hidden * mix_num) if use_norm else nn.Identity()

        self.fc_mean = nn.Linear(self.hidden, act_dim)
        self.fc_logstd = nn.Linear(self.hidden, act_dim)
        self.gate = nn.Linear(env.single_observation_space.shape[0], mix_num)

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.mix_num = mix_num
        self.topk = topk
        self.obs_len = env.single_observation_space.shape[0]
        self.expert_usage = None
        self.total_calls = 0

    def forward(self, x):
        batch = x.shape[0]
        if self.expert_usage is None:
            self.expert_usage = torch.zeros(self.mix_num, device=x.device)
        gate_scores = F.softmax(self.gate(x), dim=-1)
        topk_vals, topk_idx = torch.topk(gate_scores, k=self.topk, dim=-1)
        with torch.no_grad():
            self.total_calls += batch
            for i in range(batch):
                self.expert_usage[topk_idx[i]] += 1

        x = self.activation(self.norm1(self.fc1(x)))
        x = self.activation(self.norm2(self.fc2(x)))
        x = x.reshape(batch, self.hidden, self.mix_num)
        topk_vals = topk_vals / torch.sum(topk_vals, dim=-1, keepdim=True)
        scores = torch.zeros_like(gate_scores)
        scores[torch.arange(batch).view(-1, 1), topk_idx] = topk_vals
        scores = scores.reshape(batch, 1, -1)
        x = torch.mean(x * scores, dim=-1)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob
    
    def get_expert_usage_stats(self):
        if self.total_calls == 0:
            return torch.zeros_like(self.expert_usage)
        return self.expert_usage / self.total_calls


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__train_{task_nums2str(args.train_tasks)}__test_{task_nums2str(args.test_tasks)}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = os.path.join(args.log_dir, args.exp_name,
                            f'mix{args.mix_num}-topk{args.topk}-h{args.horizon}',
                            f"seed-{args.seed}")
    os.makedirs(log_path, exist_ok=True)
    if args.track:
        import wandb

        wandb_api_key = os.environ.get("WANDB_API_KEY", None)
        os.environ["WANDB_MODE"] = "offline"
        if wandb_api_key is not None:
            wandb.login(key=wandb_api_key)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(log_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    write2json(log_path, vars(args))
    logger = EpochLogger(log_path, seed=str(args.seed), use_tensorboard=False)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create envs
    envs = IEEE123_RL_ShmVec(
        num_envs=len(args.train_tasks),
        N_topo=5,
        N_scenerio=4,
        is_train=True,
        horizon=args.horizon,
        generator_penalty_scale=args.generator_penalty_scale,
        storage_penalty_scale=args.storage_penalty_scale,
        storage_incentive_scale=args.storage_incentive_scale,
        line_penalty_scale=args.line_penalty_scale,
    )
    test_envs = IEEE123_RL_ShmVec(
        num_envs=len(args.test_tasks),
        N_topo=5,
        N_scenerio=4,
        is_train=False,
        horizon=args.horizon,
        generator_penalty_scale=args.generator_penalty_scale,
        storage_penalty_scale=args.storage_penalty_scale,
        storage_incentive_scale=args.storage_incentive_scale,
        line_penalty_scale=args.line_penalty_scale,
    )

    args.total_timesteps = args.total_timesteps // envs.num_envs
    args.test_freq = args.test_freq // envs.num_envs
    args.save_freq = args.save_freq // envs.num_envs
    args.learning_starts = args.learning_starts // envs.num_envs

    max_action = float(envs.single_action_space.high[0])

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    envs.seed(args.seed)
    test_envs.seed(args.seed + 100)

    # Networks
    actor = Actor(envs, args.mix_num, args.topk).to(device)
    qf1 = SoftQNetwork(envs, args.mix_num, args.topk).to(device)
    qf2 = SoftQNetwork(envs, args.mix_num, args.topk).to(device)
    qf1_target = SoftQNetwork(envs, args.mix_num, args.topk).to(device)
    qf2_target = SoftQNetwork(envs, args.mix_num, args.topk).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    model = torch.nn.ModuleList([qf1, qf2, qf1_target, qf2_target, actor])

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=envs.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    # model save
    save_path = Path(os.path.join(log_path, 'model'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    def handle_signal(signal_received, frame):
        torch.save(model.state_dict(), save_path / "model_exit.pt")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, handle_signal)

    # TRY NOT TO MODIFY: start the game
    episode_returns = np.zeros(envs.num_envs)
    episode_lengths = np.zeros(envs.num_envs)
    obs, infos = envs.reset(task_nums=args.train_tasks)

    for step in range(args.total_timesteps + 1):
        global_step = step * envs.num_envs
        # ALGO LOGIC: put action logic here
        if step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions, global_step)
        dones = terminations | truncations
        rb.add(obs, next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        episode_returns += rewards
        episode_lengths += 1

        if dones.any():
            for i, task_id in enumerate(args.train_tasks):
                if dones[i]:
                    writer.add_scalar(f'train/task_{task_id}_return', episode_returns[i], global_step)
                    # writer.add_scalar(f"train/task_{task_id}_length", episode_lengths[i], global_step)
                    episode_returns[i] = 0
                    episode_lengths[i] = 0

        # ALGO LOGIC: training.
        if step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions)
            qf2_a_values = qf2(data.observations, data.actions)
            qf1_a_values = qf1_a_values.view(-1)
            qf2_a_values = qf2_a_values.view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            main_loss = (qf1_loss + qf2_loss) / 1000
            qf_loss = main_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    main_loss = ((alpha * log_pi) - min_qf_pi).mean()
                    actor_loss = main_loss

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

            if step % args.test_freq == 0:
                if args.save_model and step % args.save_freq == 0:
                    torch.save(model.state_dict(), save_path / f"model_{global_step}.pt")

                test_actor = deepcopy(actor)
                test_actor.expert_usage = None
                test_actor.total_calls = 0
                test_obs, test_info = test_envs.reset(task_nums=args.test_tasks)

                test_done = np.zeros(test_envs.num_envs, dtype=bool)
                test_return = np.zeros(test_envs.num_envs)
                test_length = np.zeros(test_envs.num_envs)

                num_tests = len(args.test_tasks)
                v_history = [[] for _ in range(num_tests)]
                p_history = [[] for _ in range(num_tests)]

                sums_list = [defaultdict(float) for _ in range(num_tests)]
                counts_list = [defaultdict(int) for _ in range(num_tests)]

                test_inference_time = 0.0
                test_inference_steps = 0

                while not np.all(test_done):
                    with torch.no_grad():
                        test_inference_start = time.time()
                        test_action, _ = test_actor.get_action(torch.Tensor(test_obs).to(device))
                        test_inference_end = time.time()
                        test_inference_time += (test_inference_end - test_inference_start)
                        test_inference_steps += 1
                        test_action = test_action.detach().cpu().numpy()

                    test_obs, test_reward, test_terminate, test_truncation, test_info = test_envs.step(test_action,
                                                                                                       global_step)
                    test_return += test_reward
                    test_length += 1
                    test_done = test_terminate | test_truncation

                    for i, single_info in enumerate(test_info):
                        v_history[i].append(float(single_info['res_V_outline']))
                        p_history[i].append(float(single_info['res_p_outline']))

                        res_p_line = single_info['res_p_line']
                        for lk, val_raw in res_p_line.items():
                            try:
                                val = abs(float(val_raw))
                            except Exception:
                                continue
                            sums_list[i][lk] += val
                            counts_list[i][lk] += 1

                    if test_done.any():
                        logger.log_tabular('charts/global_step', global_step)

                        if test_inference_steps > 0:
                            test_avg_inference_time = test_inference_time / test_inference_steps
                            writer.add_scalar("timing/avg_inference_time_per_step", test_avg_inference_time * 1000,
                                              global_step)
                            writer.add_scalar("timing/total_inference_time", test_inference_time * 1000,
                                              global_step)

                            logger.log_tabular('timing/avg_inference_time_per_step_ms',
                                               test_avg_inference_time * 1000)
                            logger.log_tabular('timing/total_inference_time_ms', test_inference_time * 1000)

                        for i, task_id in enumerate(args.test_tasks):
                            writer.add_scalar(f'test/task_{task_id}_return', test_return[i], global_step)
                            logger.log_tabular(f'test/task_{task_id}_return', test_return[i])

                            mean_v_task = float(np.mean(v_history[i])) if len(v_history[i]) > 0 else 0.0
                            mean_p_task = float(np.mean(p_history[i])) if len(p_history[i]) > 0 else 0.0
                            writer.add_scalar(f'v_outline/task_{task_id}_mean_v_outline', mean_v_task, global_step)
                            writer.add_scalar(f'p_outline/task_{task_id}_mean_p_outline', mean_p_task, global_step)
                            logger.log_tabular(f'v_outline/task_{task_id}_mean_v_outline', mean_v_task)
                            logger.log_tabular(f'p_outline/task_{task_id}_mean_p_outline', mean_p_task)

                            for lk in sorted(sums_list[i].keys()):
                                cnt = counts_list[i].get(lk, 0)
                                avg_power = float(sums_list[i][lk] / cnt) if cnt > 0 else 0.0
                                writer.add_scalar(f'lines/task_{task_id}_line_{lk}_avg_power_mw', avg_power,
                                                  global_step)

                expert_usage = test_actor.get_expert_usage_stats()
                for expert_idx in range(args.mix_num):
                    writer.add_scalar(f'expert_usage/expert_{expert_idx}', expert_usage[expert_idx].item(),
                                      global_step)
                    logger.log_tabular(f'expert_usage/expert_{expert_idx}', expert_usage[expert_idx].item())

                writer.add_scalar('expert_usage/usage_std', torch.std(expert_usage).item(), global_step)
                writer.add_scalar('expert_usage/usage_avg', torch.mean(expert_usage).item(), global_step)

                logger.log_tabular('expert_usage/usage_std', torch.std(expert_usage).item())
                logger.log_tabular('expert_usage/usage_avg', torch.mean(expert_usage).item())

                writer.add_scalar('test/overall_return', np.mean(test_return), global_step)
                logger.log_tabular('test/overall_return', np.mean(test_return))
                logger.dump_tabular()
                del test_actor

    torch.save(model.state_dict(), save_path / f"model_{args.seed}.pt")
    envs.close()
    writer.close()
    logger.close()
    if args.track:
        wandb.finish()