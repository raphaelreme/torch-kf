"""Torch-KF: Fast & Batch-friendly Kalman filtering and smoothing in PyTorch.

torch-kf provides a lightweight, GPU/CPU-accelerated implementation of the
classic (linear, Gaussian) Kalman filter, designed specifically for *many*
independent signals processed in parallel. While Kalman filtering is inherently
sequential over time and typically involves small matrices (often < 10x10 for
physics-based models), running hundreds or thousands of independent filters
becomes an ideal use case for PyTorch vectorization and hardware parallelism.

Key features
------------
- **Batched filtering**: run many independent filters without Python loops.
- **Filtering and RTS smoothing**: forward pass + Rauch-Tung-Striebel smoother.
- **Broadcast-friendly API**: most operations support broadcasting over batch
  dimensions (multiple states, measurements, and even multiple models).
- **Runs on CPU or GPU**: leverages PyTorch for multi-core CPU execution and CUDA.

Background
----------
This package is inspired by Roger R. Labbe Jr.'s `filterpy` and his companion
book *Kalman and Bayesian Filters in Python*. torch-kf currently focuses on the
traditional Kalman filter (linear dynamics + Gaussian noise).

Numerical notes
---------------
By default, torch-kf is tuned for speed: it typically runs in ``float32`` and
uses fast (but sometimes less robust) numerical schemes. If you encounter
numerical instability, consider switching to ``float64`` and enabling
``joseph_update=True`` on :class:`~torch_kf.KalmanFilter`.

Getting started
---------------
The core API consists of:
- :class:`~torch_kf.GaussianState` to represent Gaussian means/covariances.
- :class:`~torch_kf.KalmanFilter` with :meth:`~torch_kf.KalmanFilter.predict`,
  :meth:`~torch_kf.KalmanFilter.update`, :meth:`~torch_kf.KalmanFilter.filter`,
  and :meth:`~torch_kf.KalmanFilter.rts_smooth`.

Some modules provide ready-to-use motion models (e.g. constant velocity /
acceleration) that build standard ``F, H, Q, R`` matrices for you.

Notes on shapes
---------------
torch-kf uses column vectors. State and measurement vectors must have shape
``(..., dim, 1)``. Leading dimensions ``...`` are treated as batch dimensions
and may be broadcastable across operations.
"""

from .kalman_filter import GaussianState, KalmanFilter

__all__ = ["GaussianState", "KalmanFilter"]
__version__ = "0.3.1"
