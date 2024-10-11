"""This module provides helpers to build constant Kalman filters

It models the derivative of value(s) of interest up to a given order.
The highest derivative(s) are assumed to be constant.
"""

from typing import Union

import torch

from . import KalmanFilter


def interleave(x: torch.Tensor, size: int):
    """Shuffle the tensor on the first dim by group of size `size`

    >>> x = [
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5],
        [6, 6],
        [7, 7],
        [8, 8],
        [9, 9],
    ]

    >>> interleave(x, 3)
    [
        [1, 1],
        [4, 4],
        [7, 7],
        [2, 2],
        [5, 5],
        [8, 8],
        [3, 3],
        [6, 6],
        [9, 9]
    ]

    Args:
        x (Tensor): Tensor to shuffle. Expected shape: B x ...
        size (int): Size of the interleave. Should divide B (B = ks)
            Indices are mapped
            from 0, 1, 2, ..., ks -1
            to 0, s, 2s, ..., (k-1)s, 1, 1+s, ..., 1+(k-1)s, ..., s-1, s-1+s, s-1+(k-1)s
    """
    shape = list(x.shape)
    return x.reshape([-1, size] + shape[1:]).transpose(0, 1).reshape([-1] + shape[1:])


def constant_kalman_filter(  # pylint: disable=too-many-arguments
    measurement_std: Union[float, torch.Tensor],
    process_std: Union[float, torch.Tensor],
    *,
    dim=2,
    order=1,
    dt=1.0,
    expected_model=False,
    order_by_dim=False,
    approximate=False,
) -> KalmanFilter:
    """Create a constant Kalman Filter

    Create a kalman filter with the state containing values for each dimension (x, y, z, ...)
    with their derivatives up to `order`. The order-th derivatives are supposed constant during a time step.

    We consider two different models:
    The constant order-th derivative and the 0-mean (order+1)-th derivative.

    In the constant order-th derivative model, we assume that the order-th derivative is constant over a time interval
    and equals to the previous value + some gaussian noise:
    \\forall 0 < h \\le dt, x^order(t_k+h) = x^order(t_k) + w_k, where w_k \\sim N(0, process_std**2).

    In the 0-mean (order+1) derivative model, we assume that the (order+1)-th derivative is a constant
    0-mean Gaussian noise over the time interval:
    \\forall 0 < h \\le dt, x^(order + 1)(t_k+h) = w_k, w_k \\sim N(0, process_std**2)

    In both cases, the taylor expansion gives the same the process matrix. The two models only differs on the process
    noise matrix.

    Args:
        measurement_std (float | torch.Tensor): Std of the measurements.
            99.7% of measurements should fall within 3 std of the true position
            Shape: Broadcastable to dim
        process_std (float | torch.Tensor): Process noise standard deviation. With constant order-th derivative,
            it is homogenous to the order-th derivative and 99.7% of its absolute variations fall within 3 std.
            In the 0-mean (order+1)-th derivative, it is homogenous with the (order+1)-th derivative and 99.7%
            of this derivative absolute values should fall withing 3 std.
            Have look at examples to know how to fix it.
            Shape: Broadcastable to dim
        dim (int): Dimension of the motion (1d, 2d, 3d, ...)
            Default: 2
        order (int): Order of the filter (The order-th derivatives are constants)
            Default: 1 (Constant velocity)
        dt (float): Time interval
            Default: 1.0
        expected_model (bool): Use the 0-mean (order+1)-th derivative model.
            Default: False (constant order-th derivative)
        order_by_dim (bool): Order the state by dim (x, x', y, y')
            If False, order by derivatives (x, y, x', y')
            Default: False
        approximate (bool): Approximate the model at the first order.
            The noise is reduced to a single non-zero element (on the highest derivative) for each dimension.
            The value of order i is computed following x^i(t_k + h) = x^i(t_k) + h * x^(i+1)(t_k) (+ 0)
            Without approximation, it would involve the upper derivatives if modeled.
            See `create_ckf_process_matrix` and `create_ckf_process_noise`.
            Default: False

    Returns:
        torch_kf.KalmanFilter: Constant vel/acc/jerk Kalman filter
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
    """Create the process matrix (F) for the constant models

    We assume that in expectation the (order + 1)-th derivative is 0 and model the i-th derivatives up to order.
    With Taylor expansion we can simply write the i-th derivative as a weighted sum of the upper derivatives:
    x^i(t + dt) = \\sum_{k=0}^{order - i} \\frac{dt^k}{k!} x^(i + k)(t) (+ 0)

    Some examples:
    For dt=1.0 the first order process matrice is:
    F = | 1.0, 1.0 |
        | 0.0, 1.0 |

    For dt=0.5, the second order process matrix is:
    F = | 1.0, 0.5, 0.125 |
        | 0.0, 1.0, 0.5   |
        | 0.0, 0.0, 1.0   |

    Args:
        order (int): Order of the constant model. (It models derivatives up to order)
        dt (float): Time interval
            Default: 1.0
        approximate (bool): Approximate the model at the first order.
            The value of order i is computed following: x^i(t_k + h) = x^i(t_k) + h * x^(i+1)(t_k) (+ 0)
            For instance, for the second order the process matrix is now
            F = | 1.0,  dt, 0.0 |
                | 0.0, 1.0,  dt |
                | 0.0, 0.0, 1.0 |
            Default: False

    Returns:
        torch.Tensor: Process matrix
            Shape: (order + 1, order + 1)
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
    """Create the process noise matrix (Q) for the constant models

    We consider two different models:
    The constant order-th derivative and the 0-mean (order+1)-th derivative.

    In the constant order-th derivative model, we assume that the order-th derivative is constant over a time interval
    and equals to the previous value + some gaussian noise:
    x^order(t_k+h) = x^order(t_k) + w_k, w_k \\sim N(0, process_std**2).

    In the 0-mean (order+1) derivative model, we assume that the (order+1)-th derivative is a constant
    0-mean Gaussian noise over the time interval:
    x^(order + 1)(t_k+h) = w_k, w_k \\sim N(0, process_std**2)

    Args:
        process_std (float): Noise standard deviation.
            For constant order-th derivative, it is homogenous with the order-th derivative.
            For 0-mean (order+1)-th derivative, it is homoegenous with the (order+1)-th derivative.
        order (int): Order of the constant model. (It models derivatives up to order)
        dt (float): Time interval
            Default: 1.0
        expected_model (bool): Use the 0-mean (order+1)-th derivative model.
            Default: False (constant order-th derivative)
        approximate (bool): Approximate the model at the first order.
            The noise is reduced to a single non-zero element on the last row & column.
            Default: False

    Returns:
        torch.Tensor: Process noise matrix
           Shape: (order + 1, order + 1)
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
