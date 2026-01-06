"""Helpers for building constant-derivative Kalman filters.

This module provides utilities to construct the classical Kalman filter
matrices (F, H, Q, R) for *constant-derivative motion models* such as:

- constant position (order = 0),
- constant velocity (order = 1),
- constant acceleration (order = 2),
- constant jerk, etc.

The state is composed of a value and its derivatives up to a given order.
The highest-order derivative is assumed either:
- constant with additive noise, or
- driven by a zero-mean Gaussian noise on the next derivative.

These helpers are designed to integrate seamlessly with :class:`KalmanFilter`.
"""

from __future__ import annotations

import torch

from . import KalmanFilter


def interleave(x: torch.Tensor, size: int) -> torch.Tensor:
    """Interleave tensor along the first dimension.

    This utility reshuffles a tensor along its first dimension by grouping
    consecutive elements of size ``size`` and interleaving them.

    Notes:
        Indices ``0, 1, ..., k*size-1`` are remapped as:
        ``0, size, 2*size, ..., (k-1)*size,
          1, 1+size, ..., 1 + (k-1)*size,
          ...,
          size-1, 2*size-1, ..., k*size-1``

    Example:
        >>> x = torch.tensor(
        ...     [
        ...         [1, 1],
        ...         [2, 2],
        ...         [3, 3],
        ...         [4, 4],
        ...         [5, 5],
        ...         [6, 6],
        ...         [7, 7],
        ...         [8, 8],
        ...         [9, 9],
        ...     ]
        ... )
        >>> interleave(x, 3)
        tensor([
            [1, 1],
            [4, 4],
            [7, 7],
            [2, 2],
            [5, 5],
            [8, 8],
            [3, 3],
            [6, 6],
            [9, 9],
        ])

    Args:
        x (torch.Tensor): Tensor to interleave.
            Shape: ``(B, ...)``
        size: Block size used for interleaving.
            Must divide ``B`` exactly (``B = k * size``).

    Returns:
        torch.Tensor: Interleaved tensor with the same shape as ``x``.
            Shape: ``(B, ...)``

    """
    shape = list(x.shape)
    return x.reshape([-1, size, *shape[1:]]).transpose(0, 1).reshape([-1, *shape[1:]])


def constant_kalman_filter(
    measurement_std: float | torch.Tensor,
    process_std: float | torch.Tensor,
    *,
    dim=2,
    order=1,
    dt=1.0,
    expected_model=False,
    order_by_dim=False,
    approximate=False,
) -> KalmanFilter:
    r"""Create a constant-derivative Kalman filter.

    The state consists of values and their derivatives up to a given order,
    for each spatial dimension. For example:

    - ``order = 0``: position only
    - ``order = 1``: position + velocity
    - ``order = 2``: position + velocity + acceleration

    The full state dimension is ``(order + 1) * dim``.

    Two process models are supported:

    **1. Constant order-th derivative (default)**
    The highest derivative is assumed constant over a time step, with additive noise:
    \forall 0 < h \le dt, x^{(order)}(t_k+h) = x^{(order)}(t_k) + w_k, where w_k \sim N(0, process_std**2).

    **2. Zero-mean (order+1)-th derivative (expected model)**
    The (order+1)-th derivative is modeled as white Gaussian noise over the interval:
    \forall 0 < h \le dt, x^{(order + 1)}(t_k+h) = w_k, where w_k \sim N(0, process_std**2)

    In both cases, the state transition matrix ``F`` is derived from a Taylor expansion.
    The difference lies in the construction of the process noise covariance ``Q``.


    NOTE: `approximate=True` and dt=1.0 provide access to the future finite difference model defined in our KOFT paper:
          Denoting recursively dx^{i+1}(t_k) = dx^i(t_{k+1}) - dx^i(t_k) the future finite differences of order i+i,
          then the model becomes dx^order(t_{k+1}) = dx^order(t_k) + w_k, w_k \sim N(0, process_std**2).
          With approximate=True and dt=1.0, then the state is the future finite differences up to order.

    Args:
        measurement_std (float | torch.Tensor): Measurement noise standard deviation.
            Approximately 99.7% of measurements are expected to fall within
            ``±3 * measurement_std`` of the true value.
            Shape: broadcastable to ``(dim,)``.
        process_std (float | torch.Tensor): Process noise standard deviation.
            Its physical meaning depends on the chosen model:
            - constant order-th derivative: noise on the order-th derivative, where 99.7% of its variations
                should fall within ``±3 * process_std``.
            - expected model: noise on the (order+1)-th derivative. 99.7% of the (order+1)-th derivatives
                are expected to fall within ``0 ± (3 * process_std)``.
            Shape: broadcastable to ``(dim,)``.
        dim (int): Number of independent dimensions (1D, 2D, 3D, …).
            Default: 2.
        order (int): Highest derivative order included in the state (which is modeled as ~constant).
            Default: 1 (constant velocity).
        dt (float): Time step duration.
            Default: 1.0.
        expected_model (bool): If True, use the zero-mean (order+1)-th derivative model.
            Default: False.
        order_by_dim (bool): State ordering convention.
            - True: group by dimension (e.g. ``x, x', y, y'``),
            - False: group by derivative order (e.g. ``x, y, x', y'``).
            Default: False.
        approximate (bool): Use a first-order approximation of the model.
            Only the highest derivative receives process noise.
            Default: ``False``.

    Returns:
        torch_kf.KalmanFilter: Filter configured for constant velocity/acceleration/jerk models.

    """
    measurement_std = torch.broadcast_to(torch.as_tensor(measurement_std), (dim,))
    process_std = torch.broadcast_to(torch.as_tensor(process_std), (dim,))

    state_dim = (order + 1) * dim

    # Measurement model
    # We only measure the values (not the derivatives)
    # Measuremet noise is independent between the different dimensions.
    measurement_matrix = torch.eye(dim, state_dim)
    measurement_noise = torch.eye(dim) * measurement_std**2

    # Process model
    # Block matrix for each dimension
    process_matrix = torch.block_diag(*(create_ckf_process_matrix(order, dt, approximate) for _ in range(dim)))
    process_noise = torch.block_diag(
        *(create_ckf_process_noise(process_std[k].item(), order, dt, expected_model, approximate) for k in range(dim))
    )

    if order_by_dim:
        measurement_matrix = interleave(measurement_matrix.T, dim).T
    else:
        process_matrix = interleave(interleave(process_matrix, order + 1).T, order + 1).T
        process_noise = interleave(interleave(process_noise, order + 1).T, order + 1).T

    return KalmanFilter(
        process_matrix.contiguous(),
        measurement_matrix.contiguous(),
        process_noise.contiguous(),
        measurement_noise.contiguous(),
    )


def create_ckf_process_matrix(order: int, dt=1.0, approximate=False) -> torch.Tensor:
    r"""Create the process (transition) matrix ``F`` for constant-derivative models.

    The state contains derivatives up to order ``order``. Assuming the expected
    (order+1)-th derivative and above are zero, the Taylor expansion yields:

    x^{(i)}(t + dt) = \sum_{k=0}^{order - i} \frac{dt^k}{k!} x^{(i+k)}(t)

    Examples:
        - First order (constant velocity) with ``dt = 1``::

            [
                [1.0, 1.0],
                [0.0, 1.0],
            ]

        - Second order (constant acceleration) with ``dt = 0.5``::

            [
                [1, 0.5, 0.125],
                [0, 1.0, 0.5],
                [0, 0.0, 1.0],
            ]

    Args:
        order (int): Highest derivative order included in the state (which is modeled as ~constant).
        dt (float): Time step duration.
            Default: 1.0.
        approximate (bool): If True, keep only first-order terms:
            ``x^{(i)}(t+dt) = x^{(i)}(t) + dt * x^{(i+1)}(t)``.
            Higher-order Taylor terms are discarded.
            For instance, for the second order, the process matrix is now:
                [
                    [1, dt, 0],
                    [0, 1, dt],
                    [0, 0, 1],
                ]
            Default: False.

    Returns:
        torch.Tensor: Process matrix ``F``
            Shape: ``(order + 1, order + 1)``

    """
    # Compute taylors coeffs (1, dt, dt^2 / 2, ... dt^k / k!)
    range_ = torch.arange(order + 1)
    range_[0] = 1
    coefficients = torch.tensor([dt**k for k in range(order + 1)]) / range_.cumprod(0)
    if approximate:
        coefficients[2:] = 0  # Keep only 1 and dt

    # Compute the process matrix by summing diagonal tensors
    process_matrix = torch.zeros(order + 1, order + 1)
    for k, coef in enumerate(coefficients):
        process_matrix += torch.diag(torch.tensor([coef] * (order + 1 - k)), k)
    return process_matrix


def create_ckf_process_noise(
    process_std: float, order: int, dt=1.0, expected_model=False, approximate=False
) -> torch.Tensor:
    r"""Create the process noise covariance matrix ``Q``.

    Two models are supported:

    **1. Constant order-th derivative (default)**
    The highest derivative is assumed constant over a time step, with additive noise:
    \forall 0 < h \le dt, x^{(order)}(t_k+h) = x^{(order)}(t_k) + w_k, where w_k \sim N(0, process_std**2).

    **2. Zero-mean (order+1)-th derivative (expected model)**
    The (order+1)-th derivative is modeled as white Gaussian noise over the interval:
    \forall 0 < h \le dt, x^{(order + 1)}(t_k+h) = w_k, where w_k \sim N(0, process_std**2)

    The resulting covariance is obtained by integrating the noise through the
    Taylor-expanded dynamics.

    Args:
        process_std (float | torch.Tensor): Process noise standard deviation.
            - Constant model: homogeneous to the order-th derivative.
            - Expected model: homogeneous to the (order+1)-th derivative.
        order (int): Highest derivative order included in the state (which is modeled as ~constant).
        dt (float): Time step duration.
            Default: 1.0.
        expected_model (bool): Use the zero-mean (order+1)-th derivative model.
            Default: False.
        approximate (bool): If True, keep only first order terms.
            Only the highest derivative receives noise.
            Default: False.

    Returns:
        torch.Tensor: Process noise covariance matrix ``Q``.
            Shape: ``(order + 1, order + 1)``.

    """
    # Compute taylors coeffs (1, dt, dt^2 / 2, ... dt^k / k!)
    range_ = torch.arange(order + 1 + expected_model)
    range_[0] = 1
    coefficients = torch.tensor([dt**k for k in range(order + 1 + expected_model)]) / range_.cumprod(0)
    if approximate:
        coefficients[1 + expected_model :] = 0

    # For the expected model, we drop the first element (shifted by 1)
    coefficients = coefficients[expected_model:].flip(0)
    return process_std**2 * coefficients[:, None] @ coefficients[None]
