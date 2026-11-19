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


def test_steady_state_covariance_converges():
    dim_x, dim_z = 2, 2
    kf = random_kf(dim_x, dim_z)

    ssc_1 = kf.steady_state_covariance(n_iter=1)
    ssc_5 = kf.steady_state_covariance(n_iter=5)
    ssc_10 = kf.steady_state_covariance(n_iter=10)
    ssc_50 = kf.steady_state_covariance(n_iter=50)
    ssc_100 = kf.steady_state_covariance(n_iter=100)

    assert not torch.allclose(ssc_1, ssc_100)  # 1 iteration is not enough
    assert torch.linalg.norm(ssc_1 - ssc_100) >= torch.linalg.norm(ssc_5 - ssc_100)
    assert torch.linalg.norm(ssc_5 - ssc_100) >= torch.linalg.norm(ssc_10 - ssc_100)
    assert torch.linalg.norm(ssc_10 - ssc_100) >= torch.linalg.norm(ssc_50 - ssc_100)
    assert torch.allclose(ssc_10, ssc_100)  # Usually 10 (default) is enough
    assert torch.allclose(ssc_50, ssc_100)


def test_steady_state_covariance_matches_filter():
    dim_x, dim_z = 2, 2
    kf = random_kf(dim_x, dim_z)
    s = GaussianState(torch.randn(dim_x, 1), _spd_matrix(dim_x))
    measures = torch.randn(30, dim_z, 1)

    states = kf.filter(s, measures, return_all=True)
    ssc = kf.steady_state_covariance(n_iter=30)

    assert torch.allclose(states[-1].covariance, ssc)


def test_steady_state_covariance_predicted_matches_predict():
    dim_x, dim_z = 3, 2
    kf = random_kf(dim_x, dim_z)

    ssc = kf.steady_state_covariance(n_iter=30)
    ssc_predicted = kf.steady_state_covariance(n_iter=30, predicted=True)

    dummy_state = GaussianState(torch.zeros(dim_x, 1), ssc)
    expected = kf.predict(dummy_state).covariance

    assert torch.allclose(ssc_predicted, expected)


def test_steady_state_covariance_projected_matches_project():
    dim_x, dim_z = 3, 2
    kf = random_kf(dim_x, dim_z)

    ssc = kf.steady_state_covariance(n_iter=30)
    ssc_projected = kf.steady_state_covariance(n_iter=30, projected=True)

    dummy_state = GaussianState(torch.zeros(dim_x, 1), ssc)
    expected = kf.project(dummy_state).covariance

    assert torch.allclose(ssc_projected, expected)


def test_steady_state_covariance_predicted_projected_matches_predict_then_project():
    dim_x, dim_z = 4, 2
    kf = random_kf(dim_x, dim_z)

    ssc = kf.steady_state_covariance(n_iter=30)
    ssc_pred_proj = kf.steady_state_covariance(n_iter=30, predicted=True, projected=True)

    dummy_state = GaussianState(torch.zeros(dim_x, 1), ssc)
    expected = kf.project(kf.predict(dummy_state)).covariance

    assert torch.allclose(ssc_pred_proj, expected)


def test_steady_state_covariance_shape():
    dim_x, dim_z = 4, 2
    kf = random_kf(dim_x, dim_z)

    assert kf.steady_state_covariance().shape == (dim_x, dim_x)
    assert kf.steady_state_covariance(predicted=True).shape == (dim_x, dim_x)
    assert kf.steady_state_covariance(projected=True).shape == (dim_z, dim_z)
    assert kf.steady_state_covariance(predicted=True, projected=True).shape == (dim_z, dim_z)


def test_steady_state_covariance_joseph_equivalent():
    dim_x, dim_z = 3, 2
    kf = random_kf(dim_x, dim_z)

    ssc = kf.steady_state_covariance(n_iter=30)
    kf.joseph_update = True
    ssc_joseph = kf.steady_state_covariance(n_iter=30)

    assert torch.allclose(ssc, ssc_joseph)


def test_time_varying_rts_matches_joint_posterior():
    # For a linear-Gaussian chain, the RTS-smoothed marginals equal the marginals of the joint
    # posterior p(x_0, ..., x_{T-1} | z_0, ..., z_{T-1}). We check this on a scalar time-varying
    # chain where the joint posterior is available in closed form.
    one = torch.ones(1, 1, dtype=torch.float64)
    kf = KalmanFilter(one, one, one, one)  # H = R = 1, every state is observed

    process_matrix = torch.tensor([0.5, 2.0, -1.0], dtype=one.dtype).reshape(3, 1, 1)
    process_noise = torch.tensor([0.25, 1.5, 0.125], dtype=one.dtype).reshape(3, 1, 1)
    measures = torch.tensor([2.0, -1.0, 0.5, 3.0], dtype=one.dtype).reshape(4, 1, 1)

    # Forward filtering with a per-step process model, starting from x_0 ~ N(0, 1).
    state = GaussianState(torch.zeros_like(one), one)
    filtered: list[GaussianState] = []
    for t, measure in enumerate(measures):
        if t:
            state = kf.predict(state, process_matrix=process_matrix[t - 1], process_noise=process_noise[t - 1])
        state = kf.update(state, measure)
        filtered.append(state)

    trajectory = GaussianState(torch.stack([s.mean for s in filtered]), torch.stack([s.covariance for s in filtered]))

    smoothed = kf.rts_smooth(trajectory, process_matrix=process_matrix, process_noise=process_noise)

    # Closed-form joint posterior: precision = Aᵀ D⁻¹ A + I, with A = I - subdiagonal(F) the
    # innovation operator and D = diag(P_0, Q_0, ..., Q_{T-2}) the prior variances.
    innovation = torch.eye(4, dtype=one.dtype) - torch.diag(process_matrix.flatten(), -1)
    prior_variance = torch.cat([torch.ones(1, dtype=one.dtype), process_noise.flatten()])
    precision = innovation.mT @ torch.diag(1 / prior_variance) @ innovation + torch.eye(4, dtype=one.dtype)

    assert torch.allclose(smoothed.mean.flatten(), torch.linalg.solve(precision, measures.flatten()))
    assert torch.allclose(smoothed.covariance.flatten(), precision.inverse().diagonal())


def test_filter_then_rts_reduce_uncertainty():
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


def test_time_varying_filter_then_smooth_reduces_uncertainty():
    # A time-varying forward filter followed by RTS smoothing with the *same* per-step process model
    # should never increase the posterior uncertainty at any timestep.
    length = 8
    dim_x, dim_z = 3, 2
    kf = random_kf(dim_x, dim_z)
    s = GaussianState(torch.randn(dim_x, 1), _spd_matrix(dim_x))
    measures = torch.randn(length, dim_z, 1)

    # update_first (default) performs length - 1 predictions: one per transition, as RTS expects.
    process_matrix = torch.randn(length - 1, dim_x, dim_x)
    process_noise = _spd_matrix(dim_x, batch=(length - 1,))

    states = kf.filter(s, measures, return_all=True, process_matrix=process_matrix, process_noise=process_noise)
    smoothed = kf.rts_smooth(states, process_matrix=process_matrix, process_noise=process_noise)

    for t in range(length):
        assert torch.linalg.det(states[t].covariance) >= torch.linalg.det(smoothed[t].covariance)


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
