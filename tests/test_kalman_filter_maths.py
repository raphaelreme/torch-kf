"""Test mathematical concepts about KF."""

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


def test_predict_increase_uncertainty():
    dim_x, dim_z = 3, 1
    kf = random_kf(dim_x, dim_z)
    s = GaussianState(torch.randn(dim_x, 1), _spd_matrix(dim_x))

    predicted = kf.predict(s)

    assert torch.linalg.det(predicted.covariance) > torch.linalg.det(s.covariance)

    predicted_2 = kf.predict(predicted)

    assert torch.linalg.det(predicted_2.covariance) > torch.linalg.det(predicted.covariance)

    projected = kf.project(predicted)
    projected_2 = kf.project(predicted_2)

    assert projected_2.covariance.item() > projected.covariance.item()


def test_update_reduce_uncertainty():
    dim_x, dim_z = 2, 1
    kf = random_kf(dim_x, dim_z)
    s = GaussianState(torch.randn(dim_x, 1), _spd_matrix(dim_x))
    measure = torch.randn(dim_z, 1)

    updated = kf.update(s, measure)

    assert torch.linalg.det(updated.covariance) < torch.linalg.det(s.covariance)

    updated_2 = kf.update(updated, measure)

    assert torch.linalg.det(updated_2.covariance) < torch.linalg.det(updated.covariance)

    projected = kf.project(updated)
    projected_2 = kf.project(updated_2)

    assert projected_2.covariance.item() < projected.covariance.item()


def test_update_is_order_independent():
    dim_x, dim_z = 4, 2
    kf = random_kf(dim_x, dim_z)
    s = GaussianState(torch.randn(dim_x, 1), _spd_matrix(dim_x))
    measure = torch.randn(dim_z, 1)
    measure_2 = torch.randn(dim_z, 1)

    updated = kf.update(kf.update(s, measure), measure_2)
    updated_2 = kf.update(kf.update(s, measure_2), measure)

    assert torch.allclose(updated.mean, updated_2.mean)
    assert torch.allclose(updated.covariance, updated_2.covariance)


def test_several_predict_can_be_reduced_to_one():
    dim_x, dim_z = 3, 2
    kf = random_kf(dim_x, dim_z)
    s = GaussianState(torch.randn(dim_x, 1), _spd_matrix(dim_x))

    predicted = kf.predict(kf.predict(s))
    predicted_2 = kf.predict(
        s,
        process_matrix=kf.process_matrix @ kf.process_matrix,
        process_noise=kf.process_matrix @ kf.process_noise @ kf.process_matrix.mT + kf.process_noise,
    )

    assert torch.allclose(predicted.mean, predicted_2.mean)
    assert torch.allclose(predicted.covariance, predicted_2.covariance)


def test_filter_covariance_convergence():
    dim_x, dim_z = 2, 2
    kf = random_kf(dim_x, dim_z)
    s = GaussianState(torch.randn(dim_x, 1), _spd_matrix(dim_x))

    for _ in range(10):
        s = kf.predict(s)
        s = kf.update(s, torch.randn(dim_z, 1))

    covariance = s.covariance

    s = kf.predict(s)
    s = kf.update(s, torch.randn(dim_z, 1))

    assert torch.allclose(covariance, s.covariance)


def test_filter_mean_convergence_for_converged_measure():
    dim_x, dim_z = 6, 2
    kf = random_kf(dim_x, dim_z)
    s = GaussianState(torch.randn(dim_x, 1), _spd_matrix(dim_x))

    # Always the same measure, and process is identity. It should converge
    measure = torch.randn(dim_z, 1)
    kf.process_matrix = torch.eye(6)

    for _ in range(10):
        s = kf.predict(s)
        s = kf.update(s, measure)

    assert torch.allclose(kf.project(s).mean, measure)


def test_joseph_is_equivalent():
    dim_x, dim_z = 3, 3
    kf = random_kf(dim_x, dim_z)
    s = GaussianState(torch.randn(dim_x, 1), _spd_matrix(dim_x))
    measure = torch.randn(dim_z, 1)

    updated = kf.update(s, measure)
    kf.joseph_update = True
    updated_joseph = kf.update(s, measure)

    assert torch.allclose(updated.mean, updated_joseph.mean)
    assert torch.allclose(updated.covariance, updated_joseph.covariance)


def test_cholesky_is_equivalent():
    dim_x, dim_z = 4, 3
    kf = random_kf(dim_x, dim_z)
    s = GaussianState(torch.randn(dim_x, 1), _spd_matrix(dim_x))
    measure = torch.randn(dim_z, 1)

    updated = kf.update(s, measure)
    # If precision is not provided in the projection, cholesky solve is triggered
    updated_cholesky = kf.update(s, measure, projection=kf.project(s, precompute_precision=False))

    assert torch.allclose(updated.mean, updated_cholesky.mean)
    assert torch.allclose(updated.covariance, updated_cholesky.covariance)


def test_inv_t_is_contiguous_and_equivalent():
    dim_x, dim_z = 4, 3
    kf = random_kf(dim_x, dim_z)
    s = GaussianState(torch.randn(dim_x, 1), _spd_matrix(dim_x))
    dim_x, dim_z = 4, 2

    projected = kf.project(s)
    kf.inv_t = True
    projected_2 = kf.project(s)

    assert projected.precision is not None
    assert projected_2.precision is not None
    assert not projected.precision.is_contiguous()
    assert projected_2.precision.is_contiguous()
    assert torch.allclose(projected.mean, projected_2.mean)
    assert torch.allclose(projected.covariance, projected.covariance.mT)
