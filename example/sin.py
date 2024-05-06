"""Example fitlering/smoothing sinusoidal data"""

import argparse
from typing import Tuple

import matplotlib.pyplot as plt
import torch

import torch_kf

# import torch_tps

from cvkf import constant_kalman_filter


def generate_data(n: int, w0: float, noise: float, amplitude: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate sinusoidal data:

    x(t) = A sin(w0t)
    z(t) = x(t) + noise * N(0, 1)

    Args:
        n (int): Size of the sequence to generate
        w0 (float): Angular frequency
        noise (float): Gaussian noise standard deviation
        amplitude (float): Amplitude A of the sinus

    Returns:
        torch.Tensor: x(t) state of the system
            Shape: (T, 1, 1)
        torch.Tensor: z(t) measure for each state
            Shape: (T, 1, 1)
    """
    x = amplitude * torch.sin(w0 * torch.arange(n)[..., None, None])
    return x, x + noise * torch.randn_like(x)


# def smooth_tps(z: torch.Tensor, alpha=5.0):
#     t = torch.arange(z.shape[0], dtype=torch.float32)
#     z = z[..., 0]  # Shape: n, d
#     mask = ~torch.isnan(z).any(dim=-1)
#     tps = torch_tps.ThinPlateSpline(alpha)
#     return tps.fit(t[mask], z[mask]).transform(t)[..., None]


def main(order: int, n: int, measurement_std: float, amplitude: float, nans: bool):
    # Let's do 2 full periods of sinus
    w0 = 4 * torch.pi / n

    # The process errors with a constant pos/vel/acc model can be majored depending on the order usuing taylor expansion
    # | sin(w0 (t+1)) - pred_kf_order_k(sin(w0t)) | < w0^(k+1) / (k+1)!
    # Empirically as errors are not randomly distributed and this formula
    # puts to much confidence on the process (as it accumulates errors)
    # In practice, we found that using w0^k / k! worked pretty well (and a sqrt(w0) for order 0)
    process_std = amplitude * w0 ** (order + 0.5 * (order == 0)) / torch.prod(torch.arange(1, order + 1)) / 5

    if process_std < 1e-7:
        process_std = torch.tensor(1e-7)  # Prevent floating errors

    print("Parameters")
    print(f"Kalman order: {order}")
    print(f"Measurement noise: {measurement_std}")
    print(f"Process noise: {process_std}")
    print("Data: z(t) = measurement_noise * N(0, 1) + sin(w0 t)")
    print(f"Using w0={w0} for {n} points")

    kf = constant_kalman_filter(torch.tensor(measurement_std), process_std, dim=1, order=order)

    x, z = generate_data(n, w0, measurement_std, amplitude)
    if nans:
        z[n // 2 : n // 2 + n // 20] = torch.nan  # Create nan measures in the middle

    # Let's create an unkown initial state
    # Set estimation at 0, with a std of amplitude / 3
    initial_state = torch_kf.GaussianState(
        torch.zeros(kf.state_dim, 1),
        torch.eye(kf.state_dim) * torch.tensor([amplitude * w0**k * 3 for k in range(order + 1)]) ** 2,
    )

    states = kf.filter(initial_state, z, update_first=True, return_all=True)
    smoothed = kf.rts_smooth(states)
    # tps_smoothed = smooth_tps(z, measurement_std * 1000)

    print(f"Filtering MSE: {(states.mean[:, :1] - x).pow(2).mean()}")
    print(f"Smoothing MSE: {(smoothed.mean[:, :1] - x).pow(2).mean()}")
    # print(f"TPS MSE: {(tps_smoothed - x).pow(2).mean()}")

    plt.rcParams["font.size"] = 20

    plt.figure(figsize=(24, 16))
    plt.plot(x[..., 0, 0], color="k", label="True trajectory - x = A sin(w0 t)")
    plt.plot(states.mean[:, 0, 0], color="y", label="Filtered trajectory")
    plt.plot(smoothed.mean[:, 0, 0], color="g", label="Smoothed trajectory")
    # plt.plot(tps_smoothed[:, 0, 0], color="b", label="TPS trajectory")
    plt.plot(z[..., 0, 0], "o", color="r", markersize=2.0, label="Observerd trajectory - z = x + noise * N(0, 1)")

    mini = states.mean[:, 0, 0] - 3 * states.covariance[:, 0, 0].sqrt()
    maxi = states.mean[:, 0, 0] + 3 * states.covariance[:, 0, 0].sqrt()
    plt.fill_between(torch.arange(len(mini)), mini, maxi, color="y", alpha=0.5)
    mini = smoothed.mean[:, 0, 0] - 3 * smoothed.covariance[:, 0, 0].sqrt()
    maxi = smoothed.mean[:, 0, 0] + 3 * smoothed.covariance[:, 0, 0].sqrt()
    plt.fill_between(torch.arange(len(mini)), mini, maxi, color="g", alpha=0.5)

    plt.ylim(-amplitude * 1.4, amplitude * 1.4)

    plt.xlabel("t")
    plt.ylabel("x")

    plt.legend(loc="upper right")

    if order > 0:
        plt.figure(figsize=(24, 16))
        plt.plot(
            amplitude * w0 * torch.cos(w0 * torch.arange(n)), color="k", label="True velocity - v = A w0 cos(w0 t)"
        )
        plt.plot(states.mean[:, 1, 0], color="y", label="Estimated velocity (Filtering)")
        plt.plot(smoothed.mean[:, 1, 0], color="g", label="Estimated velocity (Smoothing)")

        mini = states.mean[:, 1, 0] - 3 * states.covariance[:, 1, 1].sqrt()
        maxi = states.mean[:, 1, 0] + 3 * states.covariance[:, 1, 1].sqrt()
        plt.fill_between(torch.arange(len(mini)), mini, maxi, color="y", alpha=0.5)
        mini = smoothed.mean[:, 1, 0] - 3 * smoothed.covariance[:, 1, 1].sqrt()
        maxi = smoothed.mean[:, 1, 0] + 3 * smoothed.covariance[:, 1, 1].sqrt()
        plt.fill_between(torch.arange(len(mini)), mini, maxi, color="g", alpha=0.5)

        plt.ylim(-amplitude * w0 * 1.4, amplitude * w0 * 1.4)

        plt.xlabel("t")
        plt.ylabel("v")

        plt.legend(loc="upper right")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalman filter example, filtering a noisy sinus data")
    parser.add_argument(
        "--order",
        default=2,
        type=int,
        help="Order of the kalman filter (estimate derivative up to order to predict next pos)",
    )
    parser.add_argument("--noise", default=2.0, type=float, help="Observation noise")
    parser.add_argument("--amplitude", default=20, type=int, help="Amplitude of the signal")
    parser.add_argument("--n", default=500, type=int, help="Number of points")
    parser.add_argument("--nans", action="store_true", help="Some state will not be measured")

    args = parser.parse_args()

    main(args.order, args.n, args.noise, args.amplitude, args.nans)
