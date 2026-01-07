import torch

from torch_kf import GaussianState, KalmanFilter


def _spd_matrix(dim: int, batch: tuple[int, ...] = ()) -> torch.Tensor:
    # Construct a symmetric positive definite covariance.
    cov = torch.randn(*batch, dim, dim)
    return cov @ cov.mT + 1e-2 * torch.eye(dim)


def random_kf(dim_x: int, dim_z: int, batch: tuple[int, ...] = ()) -> KalmanFilter:
    return KalmanFilter(
        torch.randn(*batch, dim_x, dim_x),
        torch.randn(*batch, dim_z, dim_x),
        _spd_matrix(dim_x, batch),
        _spd_matrix(dim_z, batch),
    )


def test_predict_broadcasts_over_models_and_states():
    models, batch = 4, 5
    dim_x, dim_z = 3, 2

    kf = random_kf(dim_x, dim_z, batch=(models, 1))  # Same models for everyone in the batch
    s = GaussianState(mean=torch.randn(batch, dim_x, 1), covariance=_spd_matrix(dim_x, batch=(batch,)))

    out = kf.predict(s)
    assert out.mean.shape == (models, batch, dim_x, 1)
    assert out.covariance.shape == (models, batch, dim_x, dim_x)

    for m in range(models):
        for b in range(batch):
            ref = kf.predict(s[b], process_matrix=kf.process_matrix[m, 0], process_noise=kf.process_noise[m, 0])
            assert torch.allclose(ref.mean, out.mean[m, b])
            assert torch.allclose(ref.covariance, out.covariance[m, b])


def test_project_broadcasts_over_models_and_states():
    models, batch = 2, 1
    dim_x, dim_z = 4, 1

    kf = random_kf(dim_x, dim_z, batch=(models, 1))  # Same models for everyone in the batch
    s = GaussianState(mean=torch.randn(batch, dim_x, 1), covariance=_spd_matrix(dim_x, batch=(batch,)))

    out = kf.project(s)
    assert out.mean.shape == (models, batch, dim_z, 1)
    assert out.covariance.shape == (models, batch, dim_z, dim_z)
    assert out.precision is not None
    assert out.precision.shape == (models, batch, dim_z, dim_z)

    for m in range(models):
        for b in range(batch):
            ref = kf.project(
                s[b], measurement_matrix=kf.measurement_matrix[m, 0], measurement_noise=kf.measurement_noise[m, 0]
            )
            assert torch.allclose(ref.mean, out.mean[m, b])
            assert torch.allclose(ref.covariance, out.covariance[m, b])
            assert ref.precision is not None
            assert torch.allclose(ref.precision, out.precision[m, b])

    # Example with a single measurement_matrix, but different measurement_noise
    kf.measurement_matrix = kf.measurement_matrix[:1]

    out = kf.project(s)
    assert out.mean.shape == (1, batch, dim_z, 1)  # Has not been expanded but still compatible
    assert out.covariance.shape == (models, batch, dim_z, dim_z)


def test_update_broadcasts_over_models_states_and_measures():
    models, batch = 5, 4
    dim_x, dim_z = 3, 2

    kf = random_kf(dim_x, dim_z, batch=(models, 1))  # Same models for everyone in the batch
    s = GaussianState(mean=torch.randn(batch, dim_x, 1), covariance=_spd_matrix(dim_x, batch=(batch,)))
    measure = torch.randn(batch, dim_z, 1)

    # Assuming measures are not aligned, one could try to update each state with each model and each measure:
    out = kf.update(s, measure[:, None, None])

    assert out.mean.shape == (batch, models, batch, dim_x, 1)
    assert out.covariance.shape == (models, batch, dim_x, dim_x)  # Not been expanded but still compatible

    for i in range(batch):
        for m in range(models):
            for b in range(batch):
                ref = kf.update(
                    s[b],
                    measure[i],
                    measurement_matrix=kf.measurement_matrix[m, 0],
                    measurement_noise=kf.measurement_noise[m, 0],
                )
                assert torch.allclose(ref.mean, out.mean[i, m, b], atol=1e-6)
                assert torch.allclose(ref.covariance, out.covariance[m, b], atol=1e-6)


def test_filter_and_smooth_broadcast_over_models_and_states():
    models, batch, length = 5, 4, 10
    dim_x, dim_z = 2, 2

    kf = random_kf(dim_x, dim_z, batch=(models, 1))  # Same models for everyone in the batch
    s = GaussianState(mean=torch.randn(batch, dim_x, 1), covariance=_spd_matrix(dim_x, batch=(batch,)))
    measures = torch.randn(length, batch, dim_z, 1)

    out = kf.filter(s, measures, return_all=True)
    out_rts = kf.rts_smooth(out)

    assert out.mean.shape == (length, models, batch, dim_x, 1)
    assert out.covariance.shape == (length, models, batch, dim_x, dim_x)

    assert out_rts.mean.shape == (length, models, batch, dim_x, 1)
    assert out_rts.covariance.shape == (length, models, batch, dim_x, dim_x)

    for m in range(models):
        for b in range(batch):
            single_kf = KalmanFilter(
                kf.process_matrix[m, 0], kf.measurement_matrix[m, 0], kf.process_noise[m, 0], kf.measurement_noise[m, 0]
            )
            ref = single_kf.filter(s[b], measures[:, b], return_all=True)
            ref_rts = single_kf.rts_smooth(ref)

            # NOTE: Due to temporal loop, we may have floating point errors that accumulates
            assert torch.allclose(ref.mean, out.mean[:, m, b], atol=1e-3)
            assert torch.allclose(ref.covariance, out.covariance[:, m, b], atol=1e-3)

            assert torch.allclose(ref_rts.mean, out_rts.mean[:, m, b], atol=1e-3)
            assert torch.allclose(ref_rts.covariance, out_rts.covariance[:, m, b], atol=1e-3)
