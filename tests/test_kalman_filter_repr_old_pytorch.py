import importlib

import torch

import torch_kf.kalman_filter


def test_repr_is_fine_with_old_pytorch():
    # Mock printoptions
    old_printoptions = None
    if hasattr(torch._tensor_str, "printoptions"):
        old_printoptions = torch._tensor_str.printoptions
        delattr(torch._tensor_str, "printoptions")

    importlib.reload(torch_kf.kalman_filter)

    process_matrix = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    process_noise = torch.eye(2) * 0.01
    measurement_matrix = torch.tensor([[1.0, 0.0]])
    measurement_noise = torch.eye(1) * 0.1
    kf = torch_kf.kalman_filter.KalmanFilter(process_matrix, measurement_matrix, process_noise, measurement_noise)

    kf_repr = str(kf)

    if old_printoptions is not None:  # RESET just in case other test depend on it.
        torch._tensor_str.printoptions = old_printoptions

    assert len(kf_repr.split("\n")) == 3 + 2 + 1
    assert kf_repr.split("\n", maxsplit=1)[0] == "Kalman Filter (State dimension: 2, Measure dimension: 1)"
    assert "Process: F = tensor([[1., 1.],   &  Q = tensor([[0.01, 0.00]," in kf_repr
    assert "Measurement: H = tensor([[1., 0.]])  &  R = tensor([[0.10]])" in kf_repr
