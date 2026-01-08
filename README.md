# Torch-KF

[![License](https://img.shields.io/github/license/raphaelreme/torch-kf)](https://github.com/raphaelreme/torch-kf/raw/main/LICENSE)
[![PyPi](https://img.shields.io/pypi/v/torch-kf)](https://pypi.org/project/torch-kf/)
[![Python](https://img.shields.io/pypi/pyversions/torch-kf)](https://pypi.org/project/torch-kf/)
[![Downloads](https://img.shields.io/pypi/dm/torch-kf)](https://pypi.org/project/torch-kf/)
[![Codecov](https://codecov.io/github/raphaelreme/torch-kf/graph/badge.svg)](https://codecov.io/github/raphaelreme/torch-kf)
[![Lint and Test](https://github.com/raphaelreme/torch-kf/actions/workflows/tests.yml/badge.svg)](https://github.com/raphaelreme/torch-kf/actions/workflows/tests.yml)

**torch-kf** is a PyTorch implementation of classic Kalman filtering and smoothing, designed for **batched processing of many independent signals**.
It supports filtering and Rauch-Tung-Striebel (RTS) smoothing, runs on CPU or GPU (via PyTorch), and natively handles batch dimensions without Python loops.

This project is inspired by Roger R. Labbe Jr.’s excellent work:
- [filterpy](https://github.com/rlabbe/filterpy)
- *Kalman and Bayesian Filters in Python* ([interactive book](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/))

Currently, torch-kf focuses on **traditional linear Kalman filters with Gaussian noise**. In the future, it may extend to a wider range of filters (e.g. EKF, UKF, IMM, ...).

---

## Why torch-kf?

Kalman filtering is inherently **sequential in time** and typically involves **small matrices** (often < 10×10 in physics-based models). As a result, a *single* Kalman filter does not benefit much from GPU acceleration and may even be faster with NumPy-based implementations such as `filterpy`.

However, many real-world problems involve **filtering large batches of independent signals in parallel**, such as:
- multi-object tracking,
- ensemble-based inference,
- large-scale simulations,
- batched time-series processing.

This is where **torch-kf** shines.

### Key ideas

- **Batch-first design**: filter hundreds or thousands of independent signals at once.
- **No Python loops** over signals: computations are vectorized.
- **Automatic parallelization**: PyTorch distributes work across multiple CPU cores or runs it on GPU.
- **Flexible broadcasting**: states, measurements, and even models can be batched.

When many signals are filtered together, torch-kf can be **orders of magnitude faster** (typically up to **200× on CPU** and **500×–1000× on GPU**) compared to running independent filters sequentially as in `filterpy`.

> [!WARNING]
> If you only need to filter a handful of signals (≈ fewer than 10), `filterpy` may still be faster due to PyTorch’s overhead on very small matrices.

---

## Numerical considerations

> [!WARNING]
> torch-kf runs in `float32` by default and prioritizes speed over maximum numerical
> robustness. It uses fast update schemes and explicit matrix inverses, which are
> well-suited for small state dimensions but can be less stable in extreme cases.
>
> If numerical stability becomes an issue, consider:
> - switching to `float64`, and
> - enabling `joseph_update=True` in `KalmanFilter`.

---

## Installation

### pip

```bash
pip install torch-kf
```

### From source

```bash
git clone git@github.com:raphaelreme/torch-kf.git  # OR https://github.com/raphaelreme/torch-kf.git
cd torch-kf
pip install .
```

---

## Getting started

```python
import torch
from torch_kf import KalmanFilter, GaussianState

# Example: filtering 100 independent 2D trajectories over 1000 timesteps
# Measurements must be column vectors (..., dim, 1)
noisy_data = torch.randn(1000, 100, 2, 1)

# Initialize the Kalman filter model
# Constant-velocity model (dt = 1)
F = torch.tensor([  # Process matrix: # x_{t+1} = x_{t} + v_{t} * dt     (dt = 1)
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], dtype=torch.float32)

Q = torch.eye(4) * 1.5**2

# Where only the position is measured, with some noise R
H = torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
], dtype=torch.float32)
R = torch.eye(2) * 3.0**2

kf = KalmanFilter(F, H, Q, R)

# Initial belief: zero position/velocity with large uncertainty
state = GaussianState(
    mean=torch.zeros(100, 4, 1),
    covariance=torch.eye(4)[None].expand(100, 4, 4) * 150**2,
)

# Filter all signals at once
states = kf.filter(
    state,
    noisy_data,
    update_first=True,
    return_all=True,
)

# states.mean:       (1000, 100, 4, 1)
# states.covariance: (1000, 100, 4, 4)

# Optional RTS smoothing
smoothed = kf.rts_smooth(states)
# smoothed.mean:       (1000, 100, 4, 1)
# smoothed.covariance: (1000, 100, 4, 4)


# Online filtering: process measure as they come
generator = ...  # Read measure from a file / sensor
for t, measure in enumerate(generator):
    # A prior on timestep t
    state = kf.predict(state)

    # Update with measure at time t
    state = kf.update(state, measure)

```

> Tip: For standard motion models (constant velocity, acceleration, jerk),
> see `torch_kf.ckf`, which provides helpers to construct well-scaled F, H, Q, and R matrices.

---

## Examples

The `examples/` folder contains simple demonstrations of constant-velocity Kalman filters (1D, 2D, …) using batched signals.

Example: filtering and smoothing noisy sinusoidal trajectories with missing (`NaN`) measurements:

![Sinusoidal position](images/sinusoidal_pos.png)
![Sinusoidal velocity](images/sinusoidal_vel.png)

We also benchmark torch-kf against `filterpy` to highlight when batched execution becomes advantageous:

![Computational time](images/computational_time.png)

For small batch sizes, PyTorch overhead dominates. As the number of signals increases, torch-kf can provide **200× speedups on CPU** and **500×+ on GPU**.

---

## Contributing

Contributions are very welcome!
Feel free to open an issue or submit a pull request.

Many extensions of Kalman filtering and smoothing are not yet implemented (e.g. variants, adaptive models). For a more feature-complete reference, see [filterpy](https://github.com/rlabbe/filterpy).

---

## Citation

This library was originally developed for large-scale object tracking in biology.
If you use torch-kf in academic work, please cite:

```bibtex
@inproceedings{reme2024particle,
  title={Particle tracking in biological images with optical-flow enhanced kalman filtering},
  author={Reme, Raphael and Newson, Alasdair and Angelini, Elsa and Olivo-Marin, Jean-Christophe and Lagache, Thibault},
  booktitle={2024 IEEE International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2024},
  organization={IEEE}
}
```
