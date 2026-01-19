# GridMoE

## Overview

GridMoE is an open-source reinforcement learning framework for large-scale power grid optimization under dynamic topologies and stochastic operating scenarios.

The framework introduces an Exogenous-Aware **Mixture-of-Experts (MoE)** policy architecture, built upon **Soft Actor-Critic (SAC)**, to enable efficient multi-task generalization across heterogeneous grid conditions.

![alt text](image.png)

This repository provides the official implementation for the paper:

> **GridMoE: Learning Exogenous-Aware MoE Policy for Power Grid Optimization under Dynamic Topologies and Scenarios**

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- Gymnasium
- Pandapower
- Numba 0.61.2 (for performance)
- Other dependencies listed in `requirements.txt` (if available)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/GridMoE.git
   cd GridMoE
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # Or manually install key packages
   pip install torch gymnasium pandapower numba stable-baselines3 wandb
   ```

3. (Optional) For GPU support, ensure CUDA is installed and compatible with PyTorch.

## Usage

### Training

To train a GridMoE agent on multiple tasks:

```bash
python sac_gridmoe.py \
  --exp_name grid_moe_mix40_topk4_h4 \
  --mix-num 40 \
  --topk 4 \
  --horizon 4 \
  --seed 0 \
  --train-tasks 0 1 ... 19 \
  --test-tasks 0 1 ... 19 \
  --total-timesteps 4000000
```

For NormMoE:

```bash
python sac_normmoe.py \
  --exp_name norm_moe_mix40_topk4_h4 \
  --mix-num 40 \
  --topk 4 \
  --horizon 4 \
  --seed 0 \
  --train-tasks 0 1 ... 19 \
  --test-tasks 0 1 ... 19 \
  --total-timesteps 4000000
```

For single-task baseline:

```bash
python sac_single.py \
  --exp_name single_task \
  --horizon 4 \
  --seed 0 \
  --train-tasks 0 0 0 0 \
  --test-tasks 0 \
  --total-timesteps 4000000
```

## Project Structure

```python
GridMoE/
├── envs/
│   ├── ieee_meta/
│   │   ├── ieee123_meta_env_v1.py    # Meta-environment for multi-task learning
│   │   ├── ieee123_rl_env_v1.py      # RL environment wrapper
│   │   ├── pandapower_build_net.py   # Grid network builder using Pandapower
│   │   ├── shmem_vec_env.py          # Shared memory vectorized environment
│   │   └── task_set/                 # Dataset for grid scenarios
├── sac_gridmoe.py                    # GridMoE implementation
├── sac_normmoe.py                    # NormMoE implementation
├── sac_single.py                     # Single-task baseline
├── utils.py                          # Utility functions
├── run_*.sh                          # Training scripts
└── README.md
```


## Citation

If you use this code in your research, please cite:

```
@article{gridmoe2026,
  title={GridMoE: Learning Exogenous-Aware Mixture-of-Experts Policy for Power Grid Optimization under Dynamic Topologies and Scenarios},
  author={Anonymous Authors},
  journal={IEEE Transactions on Industrial Informatics},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of CleanRL (https://github.com/vwxyzjn/cleanrl) for SAC implementation.
- Uses Pandapower for power system simulation.
