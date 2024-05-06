"""Example with a constant velocity kalman filter and compare with filterpy"""

import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import yaml

import filterpy.common  # type: ignore
import filterpy.kalman  # type: ignore

import torch_kf


FP_DTYPE = np.float64  # Dtype for filterpy (float64 seems slighlty faster...)


def constant_kalman_filter(
    measurement_std: torch.Tensor, process_std: torch.Tensor, dim=2, order=1
) -> torch_kf.KalmanFilter:
    """Create a constant Velocity/Acceleration/Jerk Kalman Filter

    Create a kalman filter with a state containing the positions on each dimension (x, y, z, ...)
    with their derivatives up to `order`. The order-th derivatives are supposed constant.

    Let x be the positions for each dim and x^i the i-th derivatives of these positions
    Prediction follows:
    x^i_{t+1} = x^i_t + x^{i+1}_t, for i < order
    x^order_{t+1} = x^order_t

    Args:
        measurement_std (torch.Tensor): Std of the measurements
            99.7% of measurements should fall within 3 std of the true position
            Shape: Broadcastable to dim
        process_std (torch.Tensor): Process noise, a typical value is maximum diff between two consecutive
            order-th derivative. (Eg: for constant velocity -> Maximum acceleration between two frames)
            Shape: Broadcastable to dim
        dim (int): Dimension of the motion (1d, 2d, 3d, ...)
            Default: 2
        order (int): Order of the filer (The order-th derivatives are constants)
            Default: 1 (Constant velocity)

    Returns:
        torch_kf.KalmanFilter: Constant velocity/acc/jerk Kalman filter
    """
    measurement_std = torch.broadcast_to(measurement_std, (dim,))
    process_std = torch.broadcast_to(process_std, (dim,))

    state_dim = (order + 1) * dim

    # Measurement model
    # We only measure the positions
    # Noise is independent and can have a different value in each direction
    measurement_matrix = torch.eye(dim, state_dim)
    measurement_noise = torch.eye(dim) * measurement_std**2

    # Process
    # Constant model
    # Noise in velocity estimation (which induce a noise in position estimation)
    process_matrix = torch.eye(state_dim) + torch.tensor(np.eye(state_dim, k=dim)).to(torch.float32)

    if order == 0:
        process_noise = torch.eye(state_dim) * process_std**2
    else:
        process_noise = torch.tensor(
            filterpy.common.Q_discrete_white_noise(order + 1, block_size=dim, order_by_dim=False)
        ).to(torch.float32) * torch.cat([process_std**2] * (order + 1))
    # process_noise = torch.tensor(
    #     filterpy.common.Q_discrete_white_noise(order + 1, block_size=dim, order_by_dim=False)
    # ).to(torch.float32) * torch.cat([process_std**2] * (order + 1))

    return torch_kf.KalmanFilter(process_matrix, measurement_matrix, process_noise, measurement_noise)


def convert_to_filterpy(kf: torch_kf.KalmanFilter, x0: np.ndarray, p0: np.ndarray) -> filterpy.kalman.KalmanFilter:
    """Convert a torch_kf KalmanFilter into a filterpy one

    Args:
        kf (torch_kf.KalmanFilter)
        x0 (np.ndarray): Initial state
        p0 (np.ndarray): Initial covariance

    Returns:
        filterpy.kalman.KalmanFilter
    """
    kf_fp = filterpy.kalman.KalmanFilter(dim_x=kf.state_dim, dim_z=kf.measure_dim)
    kf_fp.F = kf.process_matrix.numpy().astype(FP_DTYPE)
    kf_fp.Q = kf.process_noise.numpy().astype(FP_DTYPE)
    kf_fp.H = kf.measurement_matrix.numpy().astype(FP_DTYPE)
    kf_fp.R = kf.measurement_noise.numpy().astype(FP_DTYPE)
    kf_fp.x = x0
    kf_fp.P = p0
    kf_fp._I = kf_fp._I.astype(FP_DTYPE)  # pylint: disable=protected-access

    return kf_fp


def simulate_trajectory(
    measurement_std: float, process_std: float, n=1000, dt=1.0, batch=1, dim=1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a trajectory and its observations following a constant velocity model

    The velocity is clipped onto [-5.0, 5.0]  # A bit ugly, but just to show some examples
    """
    dims = batch, dim, 1  # Add a trailing dimension
    x, vel = torch.zeros(dims), torch.zeros(dims)
    max_vel = 5.0
    traj, measures = torch.empty((n, *dims)), torch.empty((n, *dims))
    for t in range(n):
        vel = torch.clip(vel + torch.randn(dims) * process_std, -max_vel, max_vel)
        x = x + vel * dt
        traj[t] = x
        measures[t] = x + torch.randn(dims) * measurement_std
    return traj, measures


def batch_filter_filerpy(
    kf: torch_kf.KalmanFilter,
    initial_state: torch_kf.GaussianState,
    measures: torch.Tensor,
    update_first=True,
    return_all=True,
    smooth=False,
) -> torch_kf.GaussianState:
    """Filter a batch of signal in time with filterpy

    See `torch_kf.KalmanFilter.batch_filter`
    """
    # Convert measures here (less overhead than on the fly, but more memory intensive)
    measures_np = measures.numpy().astype(FP_DTYPE)
    x0 = initial_state.mean.numpy().astype(FP_DTYPE)
    p0 = initial_state.covariance.numpy().astype(FP_DTYPE)

    if return_all:
        estimate = np.empty(
            (measures.shape[0], measures.shape[1], kf.state_dim, 1), dtype=FP_DTYPE
        )  # Save estimate (T, B, D, 1)
        cov = np.empty(
            (measures.shape[0], measures.shape[1], kf.state_dim, kf.state_dim), dtype=FP_DTYPE
        )  # And cov (T, B, D, D)

    for i in range(measures_np.shape[1]):
        # Go through the batch one by one as filterpy do not support batch computation
        # Initialize one kf by trajectory
        # There is some overhead for filterpy but this is negligible before the real computations
        kf_fp = convert_to_filterpy(kf, x0[i], p0[i])

        for t, z in enumerate(measures_np[:, i]):
            if t or not update_first:  # Do not predict on the first t
                kf_fp.predict()

            kf_fp.update(z)

            if return_all:
                estimate[t, i] = kf_fp.x
                cov[t, i] = kf_fp.P

        if return_all and smooth:
            estimate[:, i], cov[:, i], _, _ = kf_fp.rts_smoother(estimate[:, i], cov[:, i])

    if return_all:
        return torch_kf.GaussianState(
            torch.tensor(estimate, dtype=torch.float32), torch.tensor(cov, dtype=torch.float32)
        )

    return torch_kf.GaussianState(
        torch.tensor(kf_fp.x, dtype=torch.float32), torch.tensor(kf_fp.P, dtype=torch.float32)
    )


def main():
    """Check that filterpy and our code produces the same results

    And investigate the computationnal time as a function of the number of signals to filter.

    It can take from a few minutes to half an hour to run
    (You can reduce dim or batches to run a subset of computations)
    """
    process_std = 1.5
    measurement_std = 3.0
    dim = 2  # 2D
    order = 1  # CVKF
    timesteps = 100
    batches = [10**i for i in range(8)]
    smooth = False  # /!\: using True will keep everything in ram/vram and with large batches it runs out of memory

    # Create a constant velocity kalman filter
    kf = constant_kalman_filter(torch.tensor(measurement_std), torch.tensor(process_std), dim=dim, order=order)

    timings: Dict[str, List[float]] = {
        "cuda": [],
        "cpu": [],
        "filterpy": [],
    }

    for batch in tqdm.tqdm(batches):
        traj, measures = simulate_trajectory(measurement_std, process_std, n=timesteps, batch=batch, dim=dim)
        initial_state = torch_kf.GaussianState(  # Initial state with large covariance (unknown position)
            torch.zeros(batch, kf.state_dim, 1),
            torch.eye(kf.state_dim)[None].expand(batch, kf.state_dim, kf.state_dim) * 500,
        )

        t = time.time()
        if smooth:
            kf.rts_smooth(kf.filter(initial_state, measures, update_first=True, return_all=True), inplace=True)
        else:
            kf.filter(initial_state, measures, update_first=True, return_all=False)
        timings["cpu"].append(time.time() - t)

        kf = kf.to(torch.device("cuda"))

        t = time.time()
        # Initial state on measured will be converted on the fly in `filter`
        # Could convert measures here (probably faster), but requires more cuda memory.
        if smooth:
            kf.rts_smooth(kf.filter(initial_state, measures, update_first=True, return_all=True), inplace=True)
        else:
            kf.filter(initial_state, measures, update_first=True)
        timings["cuda"].append(time.time() - t)

        kf = kf.to(torch.device("cpu"))

        if batch < 10**6:
            t = time.time()
            batch_filter_filerpy(kf, initial_state, measures, update_first=True, return_all=smooth, smooth=smooth)
            timings["filterpy"].append(time.time() - t)

    print(yaml.dump(timings))

    print("Running with batch 50 and 2000 timesteps to ensure methods are equivalent")
    batch = 50
    traj, measures = simulate_trajectory(measurement_std, process_std, n=2000, batch=batch, dim=dim)
    initial_state = torch_kf.GaussianState(  # Initial state with large covariance (unknown position)
        torch.zeros(batch, kf.state_dim, 1),
        torch.eye(kf.state_dim)[None].expand(batch, kf.state_dim, kf.state_dim) * 500,
    )

    if smooth:
        state_cpu = kf.rts_smooth(kf.filter(initial_state, measures, update_first=True, return_all=True), inplace=True)
    else:
        state_cpu = kf.filter(initial_state, measures, update_first=True, return_all=True)
    state_filterpy = batch_filter_filerpy(
        kf, initial_state, measures, update_first=True, return_all=True, smooth=smooth
    )
    kf = kf.to(torch.device("cuda"))
    if smooth:
        state_cuda = kf.rts_smooth(
            kf.filter(initial_state, measures, update_first=True, return_all=True), inplace=True
        ).to(torch.device("cpu"))
    else:
        state_cuda = kf.filter(initial_state, measures, update_first=True, return_all=True).to(torch.device("cpu"))

    print(
        f"Cuda vs Cpu: Diff on mean: {(state_cuda.mean - state_cpu.mean).abs().mean()}."
        f" Diff on cov: {(state_cuda.covariance - state_cpu.covariance).abs().mean()}"
    )
    print(
        f"Cpu vs Filterpy: Diff on mean: {(state_filterpy.mean - state_cpu.mean).abs().mean()}."
        f" Diff on cov: {(state_filterpy.covariance - state_cpu.covariance).abs().mean()}"
    )

    # Plot timings
    plt.figure()
    for key, timing in timings.items():
        plt.plot(batches[: len(timing)], timing, label=key)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of filtered signal")
    plt.ylabel("Computational time")
    plt.grid()
    plt.legend()
    plt.savefig("computational_time.png")

    # Show filtering results (Plot max 2 trajectory and 100 timesteps)
    max_t = 100
    max_n = 2
    plt.figure(figsize=(24, 16))
    if dim == 2:  # Can plot in 2d
        plt.plot(traj[:max_t, :max_n, 0, 0], traj[:max_t, :max_n, 1, 0], label="True trajectory")
        plt.plot(
            measures[:max_t, :max_n, 0, 0],
            measures[:max_t, :max_n, 1, 0],
            "o",
            markersize=1.0,
            label="Observerd trajectory",
        )
        plt.plot(
            state_cpu.mean[:max_t, :max_n, 0, 0], state_cpu.mean[:max_t, :max_n, 1, 0], label="Filtered trajectory"
        )
        plt.xlabel("x")
        plt.ylabel("y")
    else:  # Simply plot the first dim
        plt.plot(traj[:max_t, :max_n, 0, 0], label="True trajectory (x)")
        plt.plot(measures[:max_t, :max_n, 0, 0], "o", markersize=1.0, label="Observerd trajectory (x)")
        plt.plot(state_cpu.mean[:max_t, :max_n, 0, 0], label="Filtered trajectory (x)")
        plt.xlabel("t")
        plt.ylabel("x")

    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
