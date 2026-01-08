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


def test_rts_reduce_uncertainty():
    length = 10
    dim_x, dim_z = 2, 1
    kf = random_kf(dim_x, dim_z)
    s = GaussianState(torch.randn(dim_x, 1), _spd_matrix(dim_x))
    measures = torch.randn(length, dim_z, 1)

    states = kf.filter(s, measures, return_all=True)
    states_rts = kf.rts_smooth(states)

    projected = kf.project(states)
    projected_rts = kf.project(states_rts)

    for t in range(length):
        assert torch.linalg.det(states[t].covariance) >= torch.linalg.det(states_rts[t].covariance)
        assert projected[t].covariance.item() >= projected_rts[t].covariance.item()


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
    assert kf_repr.split("\n")[0] == "Kalman Filter (State dimension: 2, Measure dimension: 1)"
    assert "Process: F = tensor([[1., 1.],   &  Q = tensor([[0.01, 0.00]," in kf_repr
    assert "Measurement: H = tensor([[1., 0.]])  &  R = tensor([[0.10]])" in kf_repr


def test_repr_long():
    dim_x, dim_z = 8, 8
    kf = random_kf(dim_x, dim_z)

    kf_repr = str(kf)

    assert len(kf_repr.split("\n")) == 3 + 2 * dim_x + 2 * dim_z + 2
    assert kf_repr.split("\n")[0] == f"Kalman Filter (State dimension: {dim_x}, Measure dimension: {dim_z})"
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
