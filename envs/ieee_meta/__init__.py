# from gym.envs.registration import register
from gymnasium.envs.registration import register
register(
    id='IEEE123_meta',
    entry_point='envs.ieee_meta.ieee123_meta_env_v1:IEEE123_Meta',
)