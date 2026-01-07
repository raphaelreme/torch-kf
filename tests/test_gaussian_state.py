import pytest
import torch

from torch_kf import GaussianState


def _spd_matrix(dim: int, batch: tuple[int, ...] = ()) -> torch.Tensor:
    # Construct a symmetric positive definite covariance.
    cov = torch.randn(*batch, dim, dim)
    return cov @ cov.mT + 1e-2 * torch.eye(dim)


def test_clone_is_deep_copy():
    mean = torch.randn(3, 4, 1)
    cov = _spd_matrix(4, batch=(3,))
    precision = cov.inverse()

    s = GaussianState(mean, cov, precision)
    c = s.clone()

    assert c is not s
    assert c.precision is not None
    assert s.precision is not None
    assert torch.allclose(c.mean, s.mean)
    assert torch.allclose(c.covariance, s.covariance)
    assert torch.allclose(c.precision, s.precision)

    # Mutate original, clone must not change
    s.mean.add_(1.0)
    s.covariance.mul_(2.0)
    assert not torch.allclose(c.mean, s.mean)
    assert not torch.allclose(c.covariance, s.covariance)


def test_getitem():
    s = GaussianState(torch.randn(4, 5, 2, 1), _spd_matrix(2, batch=(4, 5)))

    # Indexing
    sub = s[2]
    assert sub.mean.shape == (5, 2, 1)
    assert sub.covariance.shape == (5, 2, 2)

    sub = s[:, 3]
    assert sub.mean.shape == (4, 2, 1)
    assert sub.covariance.shape == (4, 2, 2)

    sub = s[2, 3]
    assert sub.mean.shape == (2, 1)
    assert sub.covariance.shape == (2, 2)
    assert torch.allclose(s.mean[2, 3], sub.mean)

    # Slicing
    sub = s[2:5]
    assert sub.mean.shape == (2, 5, 2, 1)
    assert sub.covariance.shape == (2, 5, 2, 2)

    sub = s[:, :2]
    assert sub.mean.shape == (4, 2, 2, 1)
    assert sub.covariance.shape == (4, 2, 2, 2)

    sub = s[1:2, -2:]
    assert sub.mean.shape == (1, 2, 2, 1)
    assert sub.covariance.shape == (1, 2, 2, 2)
    assert torch.allclose(s.mean[1:2, -2:], sub.mean)

    # Masking
    half = 0.5
    idx = torch.rand(4) > half
    sub = s[idx]
    assert sub.mean.shape == (idx.sum(), 5, 2, 1)
    assert sub.covariance.shape == (idx.sum(), 5, 2, 2)

    idx = torch.rand(5) > half
    sub = s[:, idx]
    assert sub.mean.shape == (4, idx.sum(), 2, 1)
    assert sub.covariance.shape == (4, idx.sum(), 2, 2)

    idx = torch.rand(4, 5) > half
    sub = s[idx]
    assert sub.mean.shape == (idx.sum(), 2, 1)
    assert sub.covariance.shape == (idx.sum(), 2, 2)
    assert torch.allclose(s.mean[idx], sub.mean)

    # Integer Tensor indexing
    s.precision = s.covariance.inverse()
    idx = torch.randperm(4)
    sub = s[idx]
    assert sub.mean.shape == (4, 5, 2, 1)
    assert sub.covariance.shape == (4, 5, 2, 2)
    assert torch.allclose(s.mean[idx], sub.mean)
    assert sub.precision is not None  # Test precision is also affected and indexed
    assert torch.allclose(s.precision[idx], sub.precision)


def test_setitem():
    s = GaussianState(torch.randn(5, 3, 1), _spd_matrix(3, batch=(5,)))
    s.precision = s.covariance.inverse()

    new = GaussianState(torch.ones(5, 3, 1), torch.eye(3)[None].expand(5, 3, 3))
    new.precision = new.covariance.inverse()
    new_cloned = new.clone()

    # Integer set
    s[2] = new[3]
    assert torch.allclose(s.mean[2], new.mean[3])
    assert torch.allclose(s.covariance[2], new.covariance[3])
    assert torch.allclose(s.precision[2], new.precision[3])

    # Mask set
    half = 0.5
    idx = torch.rand(5) > half
    s[idx] = new[idx]
    assert torch.allclose(s.mean[idx], new.mean[idx])
    assert torch.allclose(s.covariance[idx], new.covariance[idx])
    assert torch.allclose(s.precision[idx], new.precision[idx])

    # Non-GaussianState should error
    with pytest.raises(NotImplementedError):
        s[0] = 123  # type: ignore[assignment]

    assert torch.allclose(new.mean, new_cloned.mean)  # Nothing should change in new


def test_to_dtype():
    mean = torch.randn(1, 3, 1)
    cov = _spd_matrix(3, batch=(1,))
    s = GaussianState(mean, cov)

    s64 = s.to(torch.float64)
    assert s64.mean.dtype == torch.float64
    assert s64.covariance.dtype == torch.float64

    s64.precision = s64.covariance.inverse()

    s32 = s64.to(torch.float32)
    assert s32.mean.dtype == torch.float32
    assert s32.covariance.dtype == torch.float32
    assert s32.precision is not None
    assert s32.precision.dtype == torch.float32


def test_mahalanobis_matches_manual():
    dim = 4
    cov = _spd_matrix(dim)
    mean = torch.randn(dim, 1)
    s = GaussianState(mean, cov)

    x = torch.randn(dim, 1)
    maha = s.mahalanobis(x)

    # Manual: (x-mean)^T inv(cov) (x-mean)
    diff = mean - x
    inv = cov.inverse()
    manual = (diff.mT @ inv @ diff)[0, 0].sqrt()
    assert torch.allclose(maha, manual)


def test_mahalanobis_specific():
    mean = torch.zeros((3, 1, 1))
    measure = torch.tensor([1.0, 2.0, 10.0]).reshape(3, 1, 1)
    cov = torch.tensor([1.0, 4.0, 5.0]).reshape(3, 1, 1) ** 2

    expected = torch.tensor([1.0, 0.5, 2.0]).reshape(3)

    predicted = GaussianState(mean, cov).mahalanobis(measure)

    assert torch.allclose(predicted, expected)


def test_mahalanobis_batch_broadcasting():
    # Multiple states (B) and multiple measurements (M) broadcast.
    batch, m, dim = 3, 5, 4
    s = GaussianState(torch.randn(batch, dim, 1), _spd_matrix(dim, batch=(batch,)))
    measure = torch.randn(m, dim, 1)

    d = s[:, None].mahalanobis(measure)  # (B, 1, dim, 1) broadcasts with (M, dim, 1)

    assert d.shape == (batch, m)

    d = s.mahalanobis(measure[:, None])  # (M, 1, dim, 1) broadcasts with (B, dim, 1)

    assert d.shape == (m, batch)


def test_log_likelihood_consistency():
    s = GaussianState(torch.zeros(3, 1), _spd_matrix(3))
    x = torch.randn(3, 1)

    ll = s.log_likelihood(x)
    p = s.likelihood(x)

    assert torch.allclose(p.log(), ll, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(ll)
    assert p > 0
