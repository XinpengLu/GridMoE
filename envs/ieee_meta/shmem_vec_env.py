import numpy as np
import gymnasium as gym
import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Any, Callable, List, Tuple

from envs.ieee_meta.ieee123_rl_env_v1 import IEEE123_RL


class CloudpickleWrapper:
    def __init__(self, x: Any):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


def _np_from_rawarray(raw, dtype, shape):
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def _worker(
    idx: int,
    remote: Connection,
    parent_remote: Connection,
    env_fn_wrapped: CloudpickleWrapper,
    obs_raw,
    obs_shape: Tuple[int, ...],
    rew_raw,
    done_raw,
    trunc_raw,
):
    parent_remote.close()
    env: IEEE123_RL = env_fn_wrapped.x()

    obs_buf = _np_from_rawarray(obs_raw, np.float32, obs_shape)
    rew_buf = _np_from_rawarray(rew_raw, np.float32, (obs_shape[0],))
    done_buf = _np_from_rawarray(done_raw, np.bool_, (obs_shape[0],))
    trunc_buf = _np_from_rawarray(trunc_raw, np.bool_, (obs_shape[0],))

    last_reset_cfg = {
        "task_num": 0,
        "is_train": True,
    }

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "reset":
                (
                    task_num,
                ) = data
                last_reset_cfg = {
                    "task_num": task_num,
                }
                state, info = env.reset(
                    task_num=task_num,
                )
                obs_buf[idx, :] = state.astype(np.float32, copy=False)
                remote.send(info)
            elif cmd == "step":
                (
                    action,
                    global_step
                ) = data
                state, reward, done, tu, info = env.step(action, global_step)
                if done or tu:
                    state, _ = env.reset(**last_reset_cfg)
                obs_buf[idx, :] = state.astype(np.float32, copy=False)
                rew_buf[idx] = np.float32(reward)
                done_buf[idx] = bool(done)
                trunc_buf[idx] = bool(tu)
                remote.send(info)
            elif cmd == "seed":
                seed = data
                env.seed(seed)
                remote.send(None)
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        pass


class IEEE123_RL_ShmVec:
    def __init__(
        self,
        num_envs: int = 1,
        N_topo: int = 5,
        N_scenerio: int = 4,
        is_train: bool = True,
        horizon: int = 4,
        generator_penalty_scale: float = 30,
        storage_penalty_scale: float = 10,
        storage_incentive_scale: float = 10,
        line_penalty_scale: float = 100,
        **kwargs,
    ):
        self.num_envs = num_envs
        self.is_train = is_train
        self.horizn = horizon
        self.generator_penalty_scale = generator_penalty_scale
        self.storage_penalty_scale = storage_penalty_scale
        self.storage_incentive_scale = storage_incentive_scale
        self.line_penalty_scale = line_penalty_scale
        # build a temp env to infer spaces
        _tmp_env = IEEE123_RL(
            N_topo=N_topo,
            N_scenerio=N_scenerio,
            is_train=is_train,
            horizon=horizon,
            generator_penalty_scale = generator_penalty_scale,
            storage_penalty_scale = storage_penalty_scale,
            storage_incentive_scale = storage_incentive_scale,
            line_penalty_scale = line_penalty_scale,
            **kwargs,
        )
        self.single_observation_space = _tmp_env.observation_space
        self.single_action_space = _tmp_env.action_space
        obs_dim = int(np.prod(self.single_observation_space.shape))
        act_dim = int(np.prod(self.single_action_space.shape))
        _tmp_env.close()

        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_envs, obs_dim),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=self.single_action_space.low[0],
            high=self.single_action_space.high[0],
            shape=(self.num_envs, act_dim),
            dtype=np.float32,
        )

        self._obs_shape = (self.num_envs, obs_dim)

        # shared buffers
        self._obs_raw = mp.RawArray("f", int(np.prod(self._obs_shape)))
        self._rew_raw = mp.RawArray("f", self.num_envs)
        self._done_raw = mp.RawArray("b", self.num_envs)
        self._trunc_raw = mp.RawArray("b", self.num_envs)

        self._obs_buf = _np_from_rawarray(self._obs_raw, np.float32, self._obs_shape)
        self._rew_buf = _np_from_rawarray(self._rew_raw, np.float32, (self.num_envs,))
        self._done_buf = _np_from_rawarray(self._done_raw, np.bool_, (self.num_envs,))
        self._trunc_buf = _np_from_rawarray(self._trunc_raw, np.bool_, (self.num_envs,))

        # processes
        env_fn: Callable[[], IEEE123_RL] = lambda: IEEE123_RL(
            N_topo=N_topo,
            N_scenerio=N_scenerio,
            is_train=is_train,
            horizon=horizon,
            generator_penalty_scale=generator_penalty_scale,
            storage_penalty_scale=storage_penalty_scale,
            storage_incentive_scale=storage_incentive_scale,
            line_penalty_scale=line_penalty_scale,
            **kwargs,
        )
        env_fns = [env_fn for _ in range(self.num_envs)]

        self.remotes, self.work_remotes = zip(
            *[mp.Pipe() for _ in range(self.num_envs)]
        )
        self.ps: List[mp.Process] = []
        for i, (work_remote, remote, fn) in enumerate(
            zip(self.work_remotes, self.remotes, env_fns)
        ):
            args = (
                i,
                work_remote,
                remote,
                CloudpickleWrapper(fn),
                self._obs_raw,
                self._obs_shape,
                self._rew_raw,
                self._done_raw,
                self._trunc_raw,
            )
            p = mp.Process(target=_worker, args=args)
            p.daemon = True
            p.start()
            self.ps.append(p)
            work_remote.close()

        self.task_nums: List[int] = []

    def reset(
        self,
        task_nums: List[int],
    ):
        assert len(task_nums) == self.num_envs
        self.task_nums = list(task_nums)

        infos: List[dict] = []
        for i, remote in enumerate(self.remotes):
            remote.send(
                (
                    "reset",
                    (
                        self.task_nums[i],
                    ),
                )
            )
        for remote in self.remotes:
            infos.append(remote.recv())

        obs = self._obs_buf.copy()
        return obs, infos

    def step(self, actions: np.ndarray, global_step: int):
        assert len(actions) == self.num_envs
        for remote, action in zip(self.remotes, actions):
            data = (action, global_step)
            remote.send(("step", data))
        infos: List[dict] = []
        for remote in self.remotes:
            infos.append(remote.recv())

        obs = self._obs_buf.copy()
        rewards = self._rew_buf.copy()
        dones = self._done_buf.copy()
        truncs = self._trunc_buf.copy()

        return obs, rewards, dones, truncs, infos

    def seed(self, seed=None):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        # 一般来说，子环境要使用不同的随机种子，避免重复。
        for i, remote in enumerate(self.remotes):
            remote.send(("seed", seed + i * 100))
        for remote in self.remotes:
            remote.recv()

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for p in self.ps:
            if p.is_alive():
                p.join(timeout=0.5)
