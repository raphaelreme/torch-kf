import pytest
import torch

from torch_kf import GaussianState, KalmanFilter


def _spd_matrix(dim: int, batch: tuple[int, ...] = ()) -> torch.Tensor:
    # Construct a symmetric positive definite covariance.
    cov = torch.randn(*batch, dim, dim)
    return cov @ cov.mT + 1e-2 * torch.eye(dim)


def random_kf(dim_x: int, dim_z: int) -> KalmanFilter:
    return KalmanFilter(
        torch.randn(dim_x, dim_x),
        torch.randn(dim_z, dim_x),
        _spd_matrix(dim_x),
        _spd_matrix(dim_z),
    )


def test_filter_update_first():
    dim_x, dim_z = 2, 1
    kf = random_kf(dim_x, dim_z)
    s = GaussianState(torch.randn(dim_x, 1), _spd_matrix(dim_x))
    measures = torch.randn(1, dim_z, 1)

    out = kf.filter(s, measures, update_first=False)
    out_update_first = kf.filter(s, measures)  # update_first=True by default

    expected = kf.update(kf.predict(s), measures[0])
    expected_update_first = kf.update(s, measures[0])

    assert torch.allclose(out.mean, expected.mean)
    assert torch.allclose(out.covariance, expected.covariance)
    assert torch.allclose(out_update_first.mean, expected_update_first.mean)
    assert torch.allclose(out_update_first.covariance, expected_update_first.covariance)


def test_filter_return_all():
    batch, length = 5, 10
    dim_x, dim_z = 2, 2
    kf = random_kf(dim_x, dim_z)
    s = GaussianState(torch.randn(batch, dim_x, 1), _spd_matrix(dim_x, batch=(batch,)))
    measures = torch.randn(length, batch, dim_z, 1)

    out = kf.filter(s, measures)
    out_return_all = kf.filter(s, measures, return_all=True)

    assert out.mean.shape == (batch, dim_x, 1)
    assert out.covariance.shape == (batch, dim_x, dim_x)

    assert out_return_all.mean.shape == (length, batch, dim_x, 1)
    assert out_return_all.covariance.shape == (length, batch, dim_x, dim_x)


def test_filter_nan_skips_update_for_that_item():
    batch, length = 5, 3
    dim_x, dim_z = 2, 1
    kf = random_kf(dim_x, dim_z)
    s = GaussianState(torch.randn(batch, dim_x, 1), _spd_matrix(dim_x, batch=(batch,)))
    measures = torch.randn(length, batch, dim_z, 1)

    # For state #2, we only have a single measurement on frame 1. Updates should be skipped on frame 0 and 2.
    measures[0, 2] = torch.nan
    measures[2, 2] = torch.nan

    out = kf.filter(s, measures, return_all=True, update_first=False)

    # Recompute manually the expected states
    expected_0 = kf.predict(s[2])
    expected_2 = kf.predict(kf.update(kf.predict(kf.predict(s[2])), measures[1, 2]))

    # Expected for other states on frame 1 (tested for states #0 and #1)
    expected_1 = kf.update(kf.predict(kf.update(kf.predict(s[:2]), measures[0, :2])), measures[1, :2])

    assert torch.allclose(out[0, 2].mean, expected_0.mean)
    assert torch.allclose(out[0, 2].covariance, expected_0.covariance)

    assert torch.allclose(out[2, 2].mean, expected_2.mean, atol=1e-3)
    assert torch.allclose(out[2, 2].covariance, expected_2.covariance, atol=1e-3)

    assert torch.allclose(out[1, :2].mean, expected_1.mean)
    assert torch.allclose(out[1, :2].covariance, expected_1.covariance)


def test_filter_time_varying_model():
    length = 6
    dim_x, dim_z = 3, 2
    kf = random_kf(dim_x, dim_z).to(torch.float64)
    s = GaussianState(torch.randn(dim_x, 1, dtype=torch.float64), _spd_matrix(dim_x).to(torch.float64))
    measures = torch.randn(length, dim_z, 1, dtype=torch.float64)

    # One measurement model per update.
    measurement_matrix = torch.randn(length, dim_z, dim_x, dtype=torch.float64)
    measurement_noise = _spd_matrix(dim_z, batch=(length,)).to(torch.float64)

    # The number of predictions (hence the process time axis) depends on `update_first`.
    for update_first in (True, False):
        n_predictions = length - 1 if update_first else length
        process_matrix = torch.randn(n_predictions, dim_x, dim_x, dtype=torch.float64)
        process_noise = _spd_matrix(dim_x, batch=(n_predictions,)).to(torch.float64)

        out = kf.filter(
            s,
            measures,
            update_first=update_first,
            return_all=True,
            process_matrix=process_matrix,
            process_noise=process_noise,
            measurement_matrix=measurement_matrix,
            measurement_noise=measurement_noise,
        )

        # Reference: unroll the forward pass, slicing the same per-step matrices.
        state = s
        expected: list[GaussianState] = []
        for t, measure in enumerate(measures):
            if t or not update_first:
                i = t - 1 if update_first else t
                state = kf.predict(state, process_matrix=process_matrix[i], process_noise=process_noise[i])
            state = kf.update(
                state, measure, measurement_matrix=measurement_matrix[t], measurement_noise=measurement_noise[t]
            )
            expected.append(state)

        torch.testing.assert_close(out.mean, torch.stack([e.mean for e in expected]))
        torch.testing.assert_close(out.covariance, torch.stack([e.covariance for e in expected]))


def test_filter_constant_overrides():
    length = 5
    dim_x, dim_z = 3, 2
    kf = random_kf(dim_x, dim_z)
    s = GaussianState(torch.randn(dim_x, 1), _spd_matrix(dim_x))
    measures = torch.randn(length, dim_z, 1)

    process_matrix, process_noise = torch.randn(dim_x, dim_x), _spd_matrix(dim_x)
    measurement_matrix, measurement_noise = torch.randn(dim_z, dim_x), _spd_matrix(dim_z)

    # Plain 2D overrides (no leading time axis) are shared across every step.
    actual = kf.filter(
        s,
        measures,
        return_all=True,
        process_matrix=process_matrix,
        process_noise=process_noise,
        measurement_matrix=measurement_matrix,
        measurement_noise=measurement_noise,
    )
    expected = KalmanFilter(process_matrix, measurement_matrix, process_noise, measurement_noise).filter(
        s, measures, return_all=True
    )

    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.covariance, expected.covariance)


def test_filter_time_varying_measurement_with_nan():
    batch, length = 4, 3
    dim_x, dim_z = 2, 1
    kf = random_kf(dim_x, dim_z).to(torch.float64)
    s = GaussianState(
        torch.randn(batch, dim_x, 1, dtype=torch.float64), _spd_matrix(dim_x, batch=(batch,)).to(torch.float64)
    )
    measures = torch.randn(length, batch, dim_z, 1, dtype=torch.float64)

    # One measurement model per update (shared across the batch).
    measurement_matrix = torch.randn(length, dim_z, dim_x, dtype=torch.float64)
    measurement_noise = _spd_matrix(dim_z, batch=(length,)).to(torch.float64)

    # State #2 only sees a measurement on frame 1; updates on frame 0 and 2 are skipped.
    measures[0, 2] = torch.nan
    measures[2, 2] = torch.nan

    out = kf.filter(
        s,
        measures,
        update_first=False,
        return_all=True,
        measurement_matrix=measurement_matrix,
        measurement_noise=measurement_noise,
    )

    # Reference: unroll per item, predicting before every update and skipping NaN measures.
    expected = GaussianState(torch.empty_like(out.mean), torch.empty_like(out.covariance))
    for b in range(batch):
        state = s[b]
        for t, measure in enumerate(measures[:, b]):
            state = kf.predict(state)  # update_first=False: predict before every update
            if not torch.isnan(measure).any():
                state = kf.update(
                    state, measure, measurement_matrix=measurement_matrix[t], measurement_noise=measurement_noise[t]
                )
            expected.mean[t, b] = state.mean
            expected.covariance[t, b] = state.covariance

    torch.testing.assert_close(out.mean, expected.mean)
    torch.testing.assert_close(out.covariance, expected.covariance)


def test_filter_batched_constant_override_warns():
    one = torch.ones(1, 1)
    kf = KalmanFilter(one, one, one, one)
    length, batch = 4, 2  # 3 predictions (update_first default), != batch of 2
    s = GaussianState(torch.randn(batch, 1, 1), _spd_matrix(1, batch=(batch,)))
    measures = torch.randn(length, batch, 1, 1)
    noises = torch.tensor([0.5, 2.0]).reshape(batch, 1, 1)

    # No leading time axis (batch != predictions): treated as constant over time, with a warning.
    with pytest.warns(UserWarning, match="process_noise"):
        actual = kf.filter(s, measures, return_all=True, process_noise=noises)

    for b, noise in enumerate(noises):
        expected = KalmanFilter(one, one, noise, one).filter(s[b], measures[:, b], return_all=True)
        torch.testing.assert_close(actual.mean[:, b], expected.mean)
        torch.testing.assert_close(actual.covariance[:, b], expected.covariance)


def test_rts_time_varying_model():
    length = 6
    dim_x, dim_z = 2, 1
    kf = random_kf(dim_x, dim_z).to(torch.float64)
    process_matrix = torch.randn(length - 1, dim_x, dim_x, dtype=torch.float64)
    process_noise = _spd_matrix(dim_x, batch=(length - 1,)).to(torch.float64)

    # Build a filtered trajectory under the time-varying model.
    state = GaussianState(torch.randn(dim_x, 1, dtype=torch.float64), _spd_matrix(dim_x).to(torch.float64))
    measures = torch.randn(length, dim_z, 1, dtype=torch.float64)
    filtered: list[GaussianState] = []
    for t, measure in enumerate(measures):
        if t:
            state = kf.predict(state, process_matrix=process_matrix[t - 1], process_noise=process_noise[t - 1])
        state = kf.update(state, measure)
        filtered.append(state)
    trajectory = GaussianState(torch.stack([s.mean for s in filtered]), torch.stack([s.covariance for s in filtered]))

    smoothed = kf.rts_smooth(trajectory, process_matrix=process_matrix, process_noise=process_noise)

    # Reference: unroll the RTS backward pass, taking the prior from the public `predict`.
    expected = GaussianState(trajectory.mean.clone(), trajectory.covariance.clone())
    for t in range(length - 2, -1, -1):
        prior = kf.predict(trajectory[t], process_matrix=process_matrix[t], process_noise=process_noise[t])
        gain = trajectory.covariance[t] @ process_matrix[t].mT @ prior.covariance.inverse()
        expected.mean[t] = trajectory.mean[t] + gain @ (expected.mean[t + 1] - prior.mean)
        expected.covariance[t] = (
            trajectory.covariance[t] + gain @ (expected.covariance[t + 1] - prior.covariance) @ gain.mT
        )

    torch.testing.assert_close(smoothed.mean, expected.mean)
    torch.testing.assert_close(smoothed.covariance, expected.covariance)


def test_rts_constant_overrides():
    length = 5
    dim_x, dim_z = 3, 2
    kf = random_kf(dim_x, dim_z)
    trajectory = GaussianState(torch.randn(length, dim_x, 1), _spd_matrix(dim_x, batch=(length,)))
    process_matrix, process_noise = torch.randn(dim_x, dim_x), _spd_matrix(dim_x)

    # A plain 2D override (no leading time axis) is shared across every transition.
    expected = KalmanFilter(process_matrix, kf.measurement_matrix, process_noise, kf.measurement_noise).rts_smooth(
        trajectory
    )
    actual = kf.rts_smooth(trajectory, process_matrix=process_matrix, process_noise=process_noise, inplace=True)

    assert actual is trajectory  # inplace returns the same object
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.covariance, expected.covariance)


def test_rts_batched_constant_override():
    one = torch.ones(1, 1)
    kf = KalmanFilter(one, one, one, one)
    length, batch = 4, 2
    trajectory = GaussianState(torch.randn(length, batch, 1, 1), _spd_matrix(1, batch=(length, batch)))
    noises = torch.tensor([0.5, 2.0]).reshape(batch, 1, 1)

    # No leading time axis (batch != steps): treated as constant over time, with a warning.
    with pytest.warns(UserWarning, match="process_noise"):
        actual = kf.rts_smooth(trajectory, process_noise=noises)

    for b, noise in enumerate(noises):
        expected = KalmanFilter(one, one, noise, one).rts_smooth(trajectory[:, b])
        torch.testing.assert_close(actual.mean[:, b], expected.mean)
        torch.testing.assert_close(actual.covariance[:, b], expected.covariance)


def test_rts_override_requires_at_least_2d():
    length = 3
    dim_x, dim_z = 2, 1
    kf = random_kf(dim_x, dim_z)
    trajectory = GaussianState(torch.randn(length, dim_x, 1), _spd_matrix(dim_x, batch=(length,)))

    with pytest.raises(ValueError, match="at least two dimensions"):
        kf.rts_smooth(trajectory, process_matrix=torch.randn(dim_x))


@pytest.mark.cuda
def test_cpu_cuda_close():
    batch, length = 3, 5
    dim_x, dim_z = 2, 2
    kf = random_kf(dim_x, dim_z)
    s = GaussianState(torch.randn(batch, dim_x, 1), _spd_matrix(dim_x, batch=(batch,)))
    measures = torch.randn(length, batch, dim_z, 1)

    cpu = kf.filter(s, measures, return_all=True)
    cuda = kf.to(torch.device("cuda")).filter(s, measures, return_all=True)

    assert torch.allclose(cpu.mean, cuda.mean.cpu(), atol=1e-4)


def test_repr_short():
    process_matrix = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    process_noise = torch.eye(2) * 0.01
    measurement_matrix = torch.tensor([[1.0, 0.0]])
    measurement_noise = torch.eye(1) * 0.1
    kf = KalmanFilter(process_matrix, measurement_matrix, process_noise, measurement_noise)

    kf_repr = str(kf)

    assert len(kf_repr.split("\n")) == 3 + 2 + 1
    assert kf_repr.split("\n", maxsplit=1)[0] == "Kalman Filter (State dimension: 2, Measure dimension: 1)"
    assert "Process: F = tensor([[1., 1.],   &  Q = tensor([[0.01, 0.00]," in kf_repr
    assert "Measurement: H = tensor([[1., 0.]])  &  R = tensor([[0.10]])" in kf_repr


def test_repr_long():
    dim_x, dim_z = 8, 8
    kf = random_kf(dim_x, dim_z)

    kf_repr = str(kf)

    assert len(kf_repr.split("\n")) == 3 + 2 * dim_x + 2 * dim_z + 2
    assert kf_repr.split("\n", maxsplit=1)[0] == f"Kalman Filter (State dimension: {dim_x}, Measure dimension: {dim_z})"
    assert "Process: F = tensor(" in kf_repr
    assert "         Q = tensor(" in kf_repr
    assert "Measurement: H = tensor(" in kf_repr
    assert "             R = tensor(" in kf_repr


def test_to_convert_dtype():
    dim_x, dim_z = 3, 4
    kf = random_kf(dim_x, dim_z)

    kf64 = kf.to(torch.float64)

    assert kf64.process_matrix.dtype == torch.float64
    assert kf64.measurement_matrix.dtype == torch.float64
    assert kf64.process_noise.dtype == torch.float64
    assert kf64.measurement_noise.dtype == torch.float64

    kf32 = kf64.to(torch.float32)

    assert kf32.process_matrix.dtype == torch.float32
    assert kf32.measurement_matrix.dtype == torch.float32
    assert kf32.process_noise.dtype == torch.float32
    assert kf32.measurement_noise.dtype == torch.float32

    # And it should not affect kf64
    assert kf64.process_matrix.dtype == torch.float64
    assert kf64.measurement_matrix.dtype == torch.float64
    assert kf64.process_noise.dtype == torch.float64
    assert kf64.measurement_noise.dtype == torch.float64


# TODO: Add filtering/smoothing test with a Brownian or Directed simulator and show that it performs as expected?
