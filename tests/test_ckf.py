import pytest
import torch

from torch_kf.ckf import (
    constant_kalman_filter,
    create_ckf_process_matrix,
    create_ckf_process_noise,
    interleave,
)


def test_interleave_matches_expected():
    x = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
    y = interleave(x, 3)
    expected = torch.tensor([[1, 1], [4, 4], [7, 7], [2, 2], [5, 5], [8, 8], [3, 3], [6, 6], [9, 9]])
    assert torch.equal(y, expected)


def test_create_ckf_process_matrix_order1_dt1():
    process_matrix = create_ckf_process_matrix(order=1, dt=1.0, approximate=False)
    expected = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    assert torch.allclose(process_matrix, expected)


def test_create_ckf_process_matrix_order2_dt05():
    process_matrix = create_ckf_process_matrix(order=2, dt=0.5, approximate=False)
    expected = torch.tensor(
        [
            [1.0, 0.5, 0.125],
            [0.0, 1.0, 0.5],
            [0.0, 0.0, 1.0],
        ]
    )
    assert torch.allclose(process_matrix, expected)


def test_create_ckf_process_matrix_approximate_drops_higher_terms():
    process_matrix = create_ckf_process_matrix(order=2, dt=0.5, approximate=True)
    expected = torch.tensor(
        [
            [1.0, 0.5, 0.0],
            [0.0, 1.0, 0.5],
            [0.0, 0.0, 1.0],
        ]
    )
    assert torch.allclose(process_matrix, expected)


def test_create_ckf_process_noise_shapes_and_symmetry():
    process_noise = create_ckf_process_noise(process_std=2.0, order=2, dt=1.0, expected_model=False, approximate=False)
    assert process_noise.shape == (3, 3)
    assert torch.allclose(process_noise, process_noise.mT)

    # Eigenvalues should be >= small negative tolerance
    eig = torch.linalg.eigvalsh(process_noise)
    tol = 1e-6
    assert torch.all(eig > -tol)


def test_create_ckf_process_noise_order_3():
    process_noise = create_ckf_process_noise(process_std=1.5, order=3, dt=1.0, expected_model=False, approximate=False)

    expected = torch.tensor(
        [
            [0.0625, 0.1875, 0.3750, 0.3750],
            [0.1875, 0.5625, 1.1250, 1.1250],
            [0.3750, 1.1250, 2.2500, 2.2500],
            [0.3750, 1.1250, 2.2500, 2.2500],
        ]
    )

    assert torch.allclose(process_noise, expected)


def test_create_ckf_process_noise_expected_model():
    process_noise = create_ckf_process_noise(process_std=1.0, order=5, dt=0.5, expected_model=False, approximate=False)
    process_noise_expected = create_ckf_process_noise(
        process_std=1.0, order=5, dt=0.5, expected_model=True, approximate=False
    )

    # One can show that the expected model has an offset of 1 in the resulting noises
    assert torch.allclose(process_noise[:-1, :-1], process_noise_expected[1:, 1:])


@pytest.mark.parametrize(
    ("process_std", "order", "dt", "expected"),
    [
        (1.0, 3, 1.0, False),
        (1.0, 3, 0.5, True),
        (5.0, 2, 0.5, True),
        (0.2, 0, 2.0, False),
    ],
)
def test_create_ckf_process_noise_approximate(process_std: float, order: int, dt: float, expected: bool):
    process_noise = create_ckf_process_noise(
        process_std=process_std, order=order, dt=dt, expected_model=expected, approximate=True
    )
    assert process_noise[-1, -1] == process_std**2 * (dt**2 if expected else 1)
    process_noise[-1, -1] = 0

    assert (process_noise == 0).all()


def test_constant_kalman_filter_shapes_default_ordering():
    kf = constant_kalman_filter(
        measurement_std=3.0,
        process_std=1.5,
        dim=2,
        order=1,
        dt=1.0,
        expected_model=False,
        order_by_dim=False,
        approximate=False,
    )

    # state_dim = (order+1)*dim = 4, measure_dim=dim=2
    assert kf.process_matrix.shape == (4, 4)
    assert kf.measurement_matrix.shape == (2, 4)
    assert kf.process_noise.shape == (4, 4)
    assert kf.measurement_noise.shape == (2, 2)

    assert (
        kf.measurement_matrix
        == torch.tensor(
            [
                # x, y, dx, dy
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )
    ).all()


def test_constant_kalman_filter_order_by_dim_changes_layout_but_not_shapes():
    kf1 = constant_kalman_filter(3.0, 1.5, dim=3, order=2, order_by_dim=False)
    kf2 = constant_kalman_filter(3.0, 1.5, dim=3, order=2, order_by_dim=True)

    assert kf1.process_matrix.shape == kf2.process_matrix.shape
    assert kf1.process_noise.shape == kf2.process_noise.shape
    assert kf1.measurement_matrix.shape == kf2.measurement_matrix.shape
    assert kf1.measurement_noise.shape == kf2.measurement_noise.shape

    # Ordering differs -> at least one matrix should differ
    assert not torch.allclose(kf1.process_matrix, kf2.process_matrix)
    assert not torch.allclose(kf1.measurement_matrix, kf2.measurement_matrix)

    assert (
        kf2.measurement_matrix
        == torch.tensor(
            [
                # x,dx,ddx,y,dy,ddy,z,dz,ddz
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
            ]
        )
    ).all()
