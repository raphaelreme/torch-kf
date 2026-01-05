"""Example with constant kalman filters and compare with filterpy."""

import dataclasses
import time

import filterpy.common  # type: ignore[import-untyped]
import filterpy.kalman  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm.auto as tqdm
import yaml

import torch_kf
import torch_kf.ckf

FP_DTYPE = np.float64  # Dtype for filterpy (float64 seems slighlty faster...)


def convert_to_filterpy(kf: torch_kf.KalmanFilter, x0: np.ndarray, p0: np.ndarray) -> filterpy.kalman.KalmanFilter:
    """Convert a torch_kf KalmanFilter into a filterpy one.

    Args:
        kf (torch_kf.KalmanFilter): The kalman filter to convert
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a trajectory and its observations following a constant velocity model.

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
    """Filter a batch of signal in time with filterpy.

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


@dataclasses.dataclass
class RunTimeConfig:
    """Run time config."""

    dtype: torch.dtype = torch.float32
    device: str = "cpu"
    joseph: bool = False
    inv_t: bool = False

    def reset(self, kf: torch_kf.KalmanFilter) -> torch_kf.KalmanFilter:
        """Reset the filter with the right config."""
        kf = kf.to(torch.device(self.device)).to(self.dtype)
        kf.joseph_update = self.joseph
        kf.inv_t = self.inv_t
        return kf


def main():  # noqa: PLR0912, PLR0915
    """Check that filterpy and our code produces the same results.

    And investigate the computationnal time as a function of the number of signals to filter.

    It can take from a few minutes to half an hour to run
    (You can reduce dim or batches to run a subset of computations)
    """
    process_std = 1.5
    measurement_std = 3.0
    dim = 2  # 2D
    order = 1  # CVKF
    timesteps = 100
    batches = [10**i for i in range(7)]  # It may run out of memory for very large batches, be careful.
    smooth = False  # /!\: using True will keep everything in ram/vram and with large batches it runs out of memory

    # Create a constant velocity kalman filter
    kf = torch_kf.ckf.constant_kalman_filter(measurement_std, process_std, dim=dim, order=order)

    configs: dict[str, RunTimeConfig] = {
        "cpu32": RunTimeConfig(dtype=torch.float32, device="cpu", joseph=False, inv_t=False),
        # "cpu32-joseph": RunTimeConfig(dtype=torch.float32, device="cpu", joseph=True, inv_t=False),
        # "cpu32-invt": RunTimeConfig(dtype=torch.float32, device="cpu", joseph=False, inv_t=True),
        # "cpu32-joseph-invt": RunTimeConfig(dtype=torch.float32, device="cpu", joseph=True, inv_t=True),
        # "cpu64": RunTimeConfig(dtype=torch.float64, device="cpu", joseph=False, inv_t=False),
        "cpu64-joseph": RunTimeConfig(dtype=torch.float64, device="cpu", joseph=True, inv_t=False),
        # "cpu64-invt": RunTimeConfig(dtype=torch.float64, device="cpu", joseph=False, inv_t=True),
        # "cpu64-joseph-invt": RunTimeConfig(dtype=torch.float64, device="cpu", joseph=True, inv_t=True),
        "cuda32": RunTimeConfig(dtype=torch.float32, device="cuda", joseph=False, inv_t=False),
        # "cuda32-joseph": RunTimeConfig(dtype=torch.float32, device="cuda", joseph=True, inv_t=False),
        # "cuda32-invt": RunTimeConfig(dtype=torch.float32, device="cuda", joseph=False, inv_t=True),
        # "cuda32-joseph-invt": RunTimeConfig(dtype=torch.float32, device="cuda", joseph=True, inv_t=True),
        # "cuda64": RunTimeConfig(dtype=torch.float64, device="cuda", joseph=False, inv_t=False),
        # "cuda64-joseph": RunTimeConfig(dtype=torch.float64, device="cuda", joseph=True, inv_t=False),
        # "cuda64-invt": RunTimeConfig(dtype=torch.float64, device="cuda", joseph=False, inv_t=True),
        # "cuda64-joseph-invt": RunTimeConfig(dtype=torch.float64, device="cuda", joseph=True, inv_t=True),
    }

    timings: dict[str, list[float]] = {name: [] for name in configs}
    timings["filterpy"] = []

    for batch in tqdm.tqdm(batches):
        traj, measures = simulate_trajectory(measurement_std, process_std, n=timesteps, batch=batch, dim=dim)
        initial_state = torch_kf.GaussianState(  # Initial state with large covariance (unknown position)
            torch.zeros(batch, kf.state_dim, 1),
            torch.eye(kf.state_dim)[None].expand(batch, kf.state_dim, kf.state_dim) * 500,
        )

        for name, config in tqdm.tqdm(configs.items(), leave=False):
            kf = config.reset(kf)
            initial_state = initial_state.to(config.dtype).to(torch.device(config.device))
            measures = measures.to(config.dtype).to(torch.device(config.device))

            t = time.time()
            if smooth:
                kf.rts_smooth(kf.filter(initial_state, measures, update_first=True, return_all=True), inplace=True)
            else:
                kf.filter(initial_state, measures, update_first=True, return_all=False)
            timings[name].append(time.time() - t)

        kf = kf.to(torch.device("cpu"))
        initial_state = initial_state.to(torch.device("cpu"))
        measures = measures.to(torch.device("cpu"))

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

    kf = kf.to(torch.device("cpu")).to(torch.float64)
    kf.joseph_update = True
    kf.inv_t = False
    initial_state = initial_state.to(torch.device("cpu")).to(torch.float64)
    measures = measures.to(torch.device("cpu")).to(torch.float64)

    if smooth:
        state_cpu_p = kf.rts_smooth(
            kf.filter(initial_state, measures, update_first=True, return_all=True), inplace=True
        )
    else:
        state_cpu_p = kf.filter(initial_state, measures, update_first=True, return_all=True)

    kf = kf.to(torch.float32)
    kf.joseph_update = False
    initial_state = initial_state.to(torch.float32)
    measures = measures.to(torch.float32)

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
        f"cuda32 VS cpu32: Diff on mean: {(state_cuda.mean - state_cpu.mean).abs().mean()}."
        f" Diff on cov: {(state_cuda.covariance - state_cpu.covariance).abs().mean()}"
    )
    print(
        f"cpu64-joseph VS cpu32: Diff on mean: {(state_cpu_p.mean - state_cpu.mean).abs().mean()}."
        f" Diff on cov: {(state_cpu_p.covariance - state_cpu.covariance).abs().mean()}"
    )
    print(
        f"cpu64-joseph VS filterpy: Diff on mean: {(state_filterpy.mean - state_cpu_p.mean).abs().mean()}."
        f" Diff on cov: {(state_filterpy.covariance - state_cpu_p.covariance).abs().mean()}"
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
    if dim == 2:  # Can plot in 2d  # noqa: PLR2004
        plt.plot(traj[:max_t, :max_n, 0, 0], traj[:max_t, :max_n, 1, 0], label="True trajectory")
        plt.plot(
            measures[:max_t, :max_n, 0, 0],
            measures[:max_t, :max_n, 1, 0],
            "o",
            markersize=1.0,
            label="Observerd trajectory",
        )
        plt.plot(
            state_cpu.mean[:max_t, :max_n, 0, 0],
            state_cpu.mean[:max_t, :max_n, 1, 0],
            label=f"{'Smoothed' if smooth else 'Filtered'} trajectory",
        )
        plt.xlabel("x")
        plt.ylabel("y")
    else:  # Simply plot the first dim
        plt.plot(traj[:max_t, :max_n, 0, 0], label="True trajectory (x)")
        plt.plot(measures[:max_t, :max_n, 0, 0], "o", markersize=1.0, label="Observerd trajectory (x)")
        plt.plot(state_cpu.mean[:max_t, :max_n, 0, 0], label=f"{'Smoothed' if smooth else 'Filtered'} trajectory (x)")
        plt.xlabel("t")
        plt.ylabel("x")

    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
