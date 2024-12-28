import dataclasses
import contextlib
from typing import Optional, overload

import torch
import torch.linalg

# Note on runtime:
# Computations may be faster using cholesky decomposition and cholesky_solve
# But in real cases, dim_z is small (limiting the benefits of cholesky)
# Moreover, it is common to compute likelihood (or mahalanobis distance) after an update step which requires an
# additional cholesky solve at each call (even if the decomposition is stored),
# whereas when the inverse is computed, it can be re-used.

# pylint: disable=protected-access
if hasattr(torch._tensor_str, "printoptions"):
    printoptions = torch._tensor_str.printoptions
else:

    @contextlib.contextmanager
    def printoptions(**kwargs):
        """Change pytorch printoptions temporarily. From the future of pytorch"""
        old_printoptions = torch._tensor_str.PRINT_OPTS
        torch.set_printoptions(**kwargs)
        try:
            yield
        finally:
            torch._tensor_str.PRINT_OPTS = old_printoptions


# pylint: enable=protected-access


@dataclasses.dataclass
class GaussianState:
    """Gaussian state in Kalman Filter

    We emphasize that the mean is at least 2d (dim_x, 1).

    It also supports some of torch functionnality to clone, convert or slice both mean and covariance at once.

    Attributes:
        mean (torch.Tensor): Mean of the distribution
            Shape: (*, dim, 1)
        covariance (torch.Tensor): Covariance of the distribution
            Shape: (*, dim, dim)
        precision (Optional[torch.Tensor]): Optional inverse covariance matrix
            This may be useful for some computations (E.G mahalanobis distance, likelihood) after a predict step.
            Shape: (*, dim, dim)
    """

    mean: torch.Tensor
    covariance: torch.Tensor
    precision: Optional[torch.Tensor] = None

    def clone(self) -> "GaussianState":
        """Clone the Gaussian State using `torch.Tensor.clone`

        Returns:
            GaussianState: A copy of the Gaussian state
        """
        return GaussianState(
            self.mean.clone(), self.covariance.clone(), self.precision.clone() if self.precision is not None else None
        )

    def __getitem__(self, idx) -> "GaussianState":
        return GaussianState(
            self.mean[idx], self.covariance[idx], self.precision[idx] if self.precision is not None else None
        )

    def __setitem__(self, idx, value) -> None:
        if isinstance(value, GaussianState):
            self.mean[idx] = value.mean
            self.covariance[idx] = value.covariance
            if self.precision is not None and value.precision is not None:
                self.precision[idx] = value.precision

            return

        raise NotImplementedError()

    @overload
    def to(self, dtype: torch.dtype) -> "GaussianState": ...

    @overload
    def to(self, device: torch.device) -> "GaussianState": ...

    def to(self, fmt):
        """Convert a GaussianState to a specific device or dtype

        Args:
            fmt (torch.dtype | torch.device): Memory format to send the state to.

        Returns:
            GaussianState: The GaussianState with the right format
        """
        return GaussianState(
            self.mean.to(fmt),
            self.covariance.to(fmt),
            self.precision.to(fmt) if self.precision is not None else None,
        )

    def mahalanobis_squared(self, measure: torch.Tensor) -> torch.Tensor:
        """Computes the squared mahalanobis distance of given measure

        It supports batch computation: You can provide multiple measurements and have multiple states
        You just need to ensure that shapes are broadcastable.

        Args:
            measure (torch.Tensor): Points to consider
                Shape: (*, dim, 1)

        Returns:
            torch.Tensor: Squared mahalanobis distance for each measure/state
                Shape: (*)
        """
        diff = self.mean - measure  # You are responsible for broadcast
        if self.precision is None:
            # The inverse is transposed (back) to be contiguous: as it is symmetric
            # This is equivalent and faster to hold on the contiguous verison
            # But this may slightly increase floating errors.
            self.precision = self.covariance.inverse().mT

        return (diff.mT @ self.precision @ diff)[..., 0, 0]  # Delete trailing dimensions

    def mahalanobis(self, measure: torch.Tensor) -> torch.Tensor:
        """Computes the mahalanobis distance of given measure

        Computations of the sqrt can be slow. If you want to compare with a given threshold,
        you should rather compare the squared mahalanobis with the squared threshold.

        It supports batch computation: You can provide multiple measurements and have multiple states
        You just need to ensure that shapes are broadcastable.

        Args:
            measure (torch.Tensor): Points to consider
                Shape: (*, dim, 1)

        Returns:
            torch.Tensor: Mahalanobis distance for each measure/state
                Shape: (*)
        """
        return self.mahalanobis_squared(measure).sqrt()

    def log_likelihood(self, measure: torch.Tensor) -> torch.Tensor:
        """Computes the log-likelihood of given measure

        It supports batch computation: You can provide multiple measurements and have multiple states
        You just need to ensure that shapes are broadcastable.

        Args:
            measure (torch.Tensor): Points to consider
                Shape: (*, dim, 1)

        Returns:
            torch.Tensor: Log-likelihood for each measure/state
                Shape: (*, 1)
        """
        maha_2 = self.mahalanobis_squared(measure)
        log_det = torch.log(torch.det(self.covariance))

        return -0.5 * (self.covariance.shape[-1] * torch.log(2 * torch.tensor(torch.pi)) + log_det + maha_2)

    def likelihood(self, measure: torch.Tensor) -> torch.Tensor:
        """Computes the likelihood of given measure

        It supports batch computation: You can provide multiple measurements and have multiple states
        You just need to ensure that shapes are broadcastable.

        Args:
            measure (torch.Tensor): Points to consider
                Shape: (*, dim, 1)

        Returns:
            torch.Tensor: Likelihood for each measure/state
                Shape: (*, 1)
        """
        return self.log_likelihood(measure).exp()


class KalmanFilter:
    """Batch and fast Kalman filter implementation in PyTorch

    Kalman filtering optimally estimates the state x_k ~ N(mu_k, P_k) of a
    linear hidden markov model under Gaussian noise assumption. The model is:
    x_k = F x_{k-1} + N(0, Q)
    z_k = H x_k + N(0, R)

    where x_k is the unknown state of the system, F the state transition (or process) matrix,
    Q the process covariance, z_k the observed variables, H the measurement matrix and
    R the measurement covariance.

    .. note:

        In order to allow full flexibility on batch computation, the user has to be precise on the shape of its tensors
        1d vector should always be 2 dimensional and vertical. Check the documentation of each method.


    This is based on the numpy implementation of kalman filter: filterpy (https://filterpy.readthedocs.io/en/latest/)

    Attributes:
        process_matrix (torch.Tensor): State transition matrix (F)
            Shape: (*, dim_x, dim_x)
        measurement_matrix (torch.Tensor): Projection matrix (H)
            Shape: (*, dim_z, dim_x)
        process_noise (torch.Tensor): Uncertainty on the process (Q)
            Shape: (*, dim_x, dim_x)
        measurement_noise (torch.Tensor): Uncertainty on the measure (R)
            Shape: (*, dim_z, dim_z)
        joseph_update (bool): Use joseph update that is more numerically stable.
            Note that this increase run time by around 50%.
            Default: False

    """

    def __init__(
        self,
        process_matrix: torch.Tensor,
        measurement_matrix: torch.Tensor,
        process_noise: torch.Tensor,
        measurement_noise: torch.Tensor,
        *,
        joseph_update=False,
    ) -> None:
        # We do not check that any device/dtype/shape are shared (but they should be)
        self.process_matrix = process_matrix
        self.measurement_matrix = measurement_matrix
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self._alpha_sq = 1.0  # Memory fadding KF (As in filterpy)
        self.joseph_update = joseph_update

    @property
    def state_dim(self) -> int:
        """Dimension of the state variable"""
        return self.process_matrix.shape[0]

    @property
    def measure_dim(self) -> int:
        """Dimension of the measured variable"""
        return self.measurement_matrix.shape[0]

    @property
    def device(self) -> torch.device:
        """Device of the Kalman filter"""
        return self.process_matrix.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the Kalman filter"""
        return self.process_matrix.dtype

    @overload
    def to(self, dtype: torch.dtype) -> "KalmanFilter": ...

    @overload
    def to(self, device: torch.device) -> "KalmanFilter": ...

    def to(self, fmt):
        """Convert a Kalman filter to a specific device or dtype

        Args:
            fmt (torch.dtype | torch.device): Memory format to send the filter to.

        Returns:
            KalmanFilter: The filter with the right format
        """
        return KalmanFilter(
            self.process_matrix.to(fmt),
            self.measurement_matrix.to(fmt),
            self.process_noise.to(fmt),
            self.measurement_noise.to(fmt),
        )

    def predict(
        self,
        state: GaussianState,
        *,
        process_matrix: Optional[torch.Tensor] = None,
        process_noise: Optional[torch.Tensor] = None,
    ) -> GaussianState:
        """Prediction from the given state

        Use the process model x_{k+1} = F x_k + N(0, Q) to compute the prior on the future state.
        Support batch computation: you can provide multiple models (F, Q) or/and multiple states.
        You just need to ensure that shapes are broadcastable.

        Example:
            # Initialize a random batch of gaussian state (5d)
            state = GaussianState(
                torch.randn(50, 5, 1),  # The last dimension is required.
                torch.randn(50, 5, 5),
            )

            # Use a single process model
            process_matrix = torch.randn(5, 5)  # Compatible with (50, 5, 5)
            process_noise = torch.randn(5, 5)

            predicted = kf.predict(state, process_matrix, process_noise)
            predicted.mean  # Shape: (50, 5, 1)  # Predictions for each state
            predicted.covariance  # Shape: (50, 5, 5)

            # Use several models
            process_matrix = torch.randn(10, 1, 5, 5)  # Compatible with (50, 5, 5)
            process_noise = torch.randn(1, 1, 5, 5)  # Let's use the same noise for each process matrix

            predicted = kf.predict(state, process_matrix, process_noise)
            predicted.mean  # Shape: (10, 50, 5, 1)  # Predictions for each model and state
            predicted.covariance  # Shape: (10, 50, 5, 5)

        Args:
            state (GaussianState): Current state estimation. Should have dim_x dimension.
            process_matrix (Optional[torch.Tensor]): Overwrite the default transition matrix
                Shape: (*, dim_x, dim_x)
            process_noise (Optional[torch.Tensor]): Overwrite the default process noise)
                Shape: (*, dim_x, dim_x)

        Returns:
            GaussianState: Prior on the next state. Will have dim_x dimension.

        """
        if process_matrix is None:
            process_matrix = self.process_matrix
        if process_noise is None:
            process_noise = self.process_noise

        mean = process_matrix @ state.mean
        covariance = self._alpha_sq * process_matrix @ state.covariance @ process_matrix.mT + process_noise

        return GaussianState(mean, covariance)

    def project(
        self,
        state: GaussianState,
        *,
        measurement_matrix: Optional[torch.Tensor] = None,
        measurement_noise: Optional[torch.Tensor] = None,
        precompute_precision=True,
    ) -> GaussianState:
        """Project the current state (usually the prior) onto the measurement space

        Use the measurement equation: z_k = H x_k + N(0, R).
        Support batch computation: You can provide multiple measurements, projections models (H, R)
        or/and multiple states. You just need to ensure that shapes are broadcastable.

        Example:
            # Initialize a random batch of gaussian state (5d)
            state = GaussianState(
                torch.randn(50, 5, 1),  # The last dimension is required.
                torch.randn(50, 5, 5),
            )

            # Use a single projection model
            measurement_matrix = torch.randn(3, 5)  # Compatible with (50, 5, 5)
            measurement_noise = torch.randn(3, 3)  # Broadcastable with (50, 3, 3)

            projection = kf.project(state, measurement_matrix, measurement_noise)
            projection.mean  # Shape: (50, 3, 1)  # projection for each state
            projection.covariance  # Shape: (50, 3, 3)

            # Use several models
            measurement_matrix = torch.randn(1, 1, 3, 5)  # Same measurement for each model, compatible with (50, 5, 5)
            measurement_noise = torch.randn(10, 1, 3, 3)  # Use different noises

            projection = kf.project(state, measurement_matrix, measurement_noise)
            projection.mean  # Shape: (1, 50, 3, 1)  # /!\\, the state will not be broadcasted to (10, 50, 5, 1).
            projection.covariance  # Shape: (10, 50, 3, 3)  # Projection cov for each model and each state

        Args:
            state (GaussianState): Current state estimation (Usually the results of `predict`)
            measurement_matrix (Optional[torch.Tensor]): Overwrite the default projection matrix
                Shape: (*, dim_z, dim_x)
            measurement_noise (Optional[torch.Tensor]): Overwrite the default projection noise)
                Shape: (*, dim_z, dim_z)
            precompute_precision (bool): Precompute precision matrix (inverse covariance)
                Done once to prevent more computations
                Default: True

        Returns:
            GaussianState: Prior on the next state

        """
        if measurement_matrix is None:
            measurement_matrix = self.measurement_matrix
        if measurement_noise is None:
            measurement_noise = self.measurement_noise

        mean = measurement_matrix @ state.mean
        covariance = measurement_matrix @ state.covariance @ measurement_matrix.mT + measurement_noise

        return GaussianState(
            mean,
            covariance,
            (
                # Cholesky inverse is usually slower with small dimensions
                # The inverse is transposed (back) to be contiguous: as it is symmetric
                # This is equivalent and faster to hold on the contiguous verison
                # But this may slightly increase floating errors.
                covariance.inverse().mT
                if precompute_precision
                else None
            ),
        )

    def update(
        self,
        state: GaussianState,
        measure: torch.Tensor,
        *,
        projection: Optional[GaussianState] = None,
        measurement_matrix: Optional[torch.Tensor] = None,
        measurement_noise: Optional[torch.Tensor] = None,
    ) -> GaussianState:
        """Compute the posterior estimation by integrating a new measure into the state

        Support batch computation: You can provide multiple measurements, projections models (H, R)
        or/and multiple states. You just need to ensure that shapes are broadcastable.

        Example:
            # Initialize a random batch of gaussian state (5d)
            state = GaussianState(
                torch.randn(50, 5, 1),  # The last dimension is required.
                torch.randn(50, 5, 5),
            )

            # Use a single projection model and a single measurement for each state
            measurement_matrix = torch.randn(3, 5)  # Compatible with (50, 5, 5)
            measurement_noise = torch.randn(3, 3)  # Broadcastable with (50, 3, 3)
            measure = torch.randn(50, 3, 3)

            new_state = kf.update(state, measure, None, measurement_matrix, measurement_noise)
            new_state.mean  # Shape: (50, 5, 1)  # Each state has been updated
            new_state.covariance  # Shape: (50, 5, 5)

            # Use several models and a single measurement for each state
            measurement_matrix = torch.randn(10, 1, 3, 5)  # Compatible with (50, 5, 5)
            measurement_noise = torch.randn(10, 1, 3, 3)  # Use different noises
            measure = torch.randn(50, 3, 1)  # The last unsqueezed dimension is required

            new_state = kf.update(state, measure, None, measurement_matrix, measurement_noise)
            new_state.mean  # Shape: (10, 50, 5, 1)  # Each state for each model has been updated
            new_state.covariance  # Shape: (10, 50, 5, 1)

            # Use several models and all measurements for each state
            measurement_matrix = torch.randn(10, 1, 3, 5)  # Compatible with (50, 5, 5)
            measurement_noise = torch.randn(10, 1, 3, 3)  # Use different noises
            # We have 50 measurements and we update each state/model with every measurements
            measure = torch.randn(50, 1, 1, 3, 1)

            new_state = kf.update(state, measure, None, measurement_matrix, measurement_noise)
            new_state.mean  # Shape: (50, 10, 50, 5, 1)  # Update for each measure, model and previous state
            new_state.covariance  # Shape: (10, 50, 5, 5)  # /!\\ The cov is not broadcasted to (50, 10, 50, 5, 5)

        Args:
            state (GaussianState): Current state estimation (Usually the results of `predict`)
            measure (torch.Tensor): State measure (z_k) (The last unsqueezed dimension is required)
                Shape: (*, dim_z, 1)
            projection (Optional[GaussianState]): Precomputed projection if any.
            measurement_matrix (Optional[torch.Tensor]): Overwrite the default projection matrix
                Shape: (*, dim_z, dim_x)
            measurement_noise (Optional[torch.Tensor]): Overwrite the default projection noise)
                Shape: (*, dim_z, dim_z)

        Returns:
            GaussianState: Prior on the next state

        """
        if measurement_matrix is None:
            measurement_matrix = self.measurement_matrix
        if measurement_noise is None:
            measurement_noise = self.measurement_noise
        if projection is None:
            projection = self.project(state, measurement_matrix=measurement_matrix, measurement_noise=measurement_noise)

        residual = measure - projection.mean

        if projection.precision is None:  # Old version using cholesky and solve to prevent the inverse computation
            # Find K without inversing S but by solving the linear system SK^T = (PH^T)^T
            # May be slightly more robust but is usually slower in low dimension
            chol_decomposition, _ = torch.linalg.cholesky_ex(projection.covariance)  # pylint: disable=not-callable
            kalman_gain = torch.cholesky_solve(measurement_matrix @ state.covariance.mT, chol_decomposition).mT
        else:
            kalman_gain = state.covariance @ measurement_matrix.mT @ projection.precision

        mean = state.mean + kalman_gain @ residual

        if self.joseph_update:
            factor = torch.eye(self.state_dim, dtype=self.dtype, device=self.device) - kalman_gain @ measurement_matrix
            covariance = factor @ state.covariance @ factor.mT + kalman_gain @ measurement_noise @ kalman_gain.mT
        else:
            covariance = state.covariance - kalman_gain @ measurement_matrix @ state.covariance

        return GaussianState(mean, covariance)

    def filter(
        self, state: GaussianState, measures: torch.Tensor, update_first=False, return_all=False
    ) -> GaussianState:
        """Filter signals with given measures

        It handles most of the default use-cases but it remains very standard, you probably will have to rewrite
        it for a specific problem. It supports nan values in measures. The states associated with a nan measure
        are not updated. For a multidimensional measure, a single nan value will invalidate all the measure
        (because the measurement matrix cannot be infered).

        Limitations examples:
        It only works if states and measures are already aligned (associated).
        It is memory intensive as it requires the input (and output if `return_all`) to be stored in a tensor.
        It does not support changing the Kalman model (F, Q, H, R) in time.

        Again all of this can be done manually using this function as a baseline for a more precise code.

        Args:
            state (GaussianState): Initial state to start filtering from
            measures (torch.Tensor): Measures in time
                Shape: (T, *, dim_z, 1)
            update_first (bool): Only update for the first timestep, then goes back to the predict / update cycle.
                Default: False
            return_all (bool): The state returns contains all states after an update step.
                To access predicted states, you either have to run again `predict` on the result, or do it manually.
                Default: False (Returns only the last state)

        Returns:
            GaussianState: The final updated state or all the update states (in a single GaussianState object)
        """
        # Convert state to the right dtype and device
        state = state.to(self.dtype).to(self.device)

        saver: GaussianState

        for t, measure in enumerate(measures):
            if t or not update_first:  # Do not predict on the first t
                state = self.predict(state)

            # Convert on the fly the measure to avoid to store them all in cuda memory
            # To avoid this overhead, the conversion can be done by the user before calling `batch_filter`
            measure = measure.to(self.dtype).to(self.device, non_blocking=True)

            # Support for nan measure: Do not update state associated with a nan measure
            mask = torch.isnan(measure[..., 0]).any(dim=-1)
            if mask.any():
                valid_state = GaussianState(state.mean[~mask], state.covariance[~mask])
                valid_state = self.update(valid_state, measure[~mask])  # Update states with valid measures
                state.mean[~mask] = valid_state.mean
                state.covariance[~mask] = valid_state.covariance
            else:
                state = self.update(state, measure)  # Update states

            if return_all:
                if t == 0:  # Create the saver now that we know the size of an updated state
                    # In this implementation, it cannot evolve in time, but it still supports
                    # to have a change from the initial_state shape to the first updated state (with the first measure)
                    saver = GaussianState(
                        torch.empty(
                            (measures.shape[0], *state.mean.shape), dtype=state.mean.dtype, device=state.mean.device
                        ),
                        torch.empty(
                            (measures.shape[0], *state.covariance.shape),
                            dtype=state.mean.dtype,
                            device=state.mean.device,
                        ),
                    )

                saver.mean[t] = state.mean
                saver.covariance[t] = state.covariance

        if return_all:
            return saver

        return state

    def rts_smooth(self, state: GaussianState, inplace=False) -> GaussianState:
        """Smooth filtered signals using Rauch-Tung-Striebel smoothing.

        It handles most of the default use-cases but it remains very standard, you may have to rewrite it for
        a specific smoothing problem. For instance it does not support changing the Kalman model (F, Q, H, R) in time.

        Smoothing can be done manually using this function as a baseline for a more precise code.

        Args:
            state (GaussianState): All filtered states in time. (Typically returned by filter with `return_all=True`)
                The first dimension of the state is seen as time: for instance mean is expected
                to be (T, *, dim_x, 1)
            inplace (bool): Modify inplace state
                Default: False (will allocate twice more memory)

        Returns:
            GaussianState: All smoothed states in time
                The first dimension is time: for instance the covariance is (T, *, dim_x, dim_x)
        """
        if inplace:
            out = state
        else:
            out = GaussianState(
                state.mean.clone(),
                state.covariance.clone(),
            )

        # Iterate backward to update all states (except the last one which is already fine)
        for t in range(state.mean.shape[0] - 2, -1, -1):
            cov_at_process = state.covariance[t] @ self.process_matrix.mT
            predicted_covariance = self.process_matrix @ cov_at_process + self.process_noise

            kalman_gain = cov_at_process @ predicted_covariance.inverse().mT
            out.mean[t] += kalman_gain @ (out.mean[t + 1] - self.process_matrix @ state.mean[t])
            out.covariance[t] += kalman_gain @ (out.covariance[t + 1] - predicted_covariance) @ kalman_gain.mT

        return out

    def __repr__(self) -> str:
        header = f"Kalman Filter (State dimension: {self.state_dim}, Measure dimension: {self.measure_dim})"

        # Process string
        with printoptions(profile="short", sci_mode=False, linewidth=80):
            process_matrix_repr = str(self.process_matrix).split("\n")
            process_noise_repr = str(self.process_noise).split("\n")

        max_char_matrix = max((len(line) for line in process_matrix_repr))
        max_char_noise = max((len(line) for line in process_noise_repr))
        if max_char_matrix + max_char_noise <= 110:  # Single line
            process_matrix_repr = [line + " " * (max_char_matrix - len(line)) for line in process_matrix_repr]

            process_header = ["Process: F = "] + ["             "] * (len(process_matrix_repr) - 1)
            process_sep = ["  &  Q = "] + ["         "] * (len(process_matrix_repr) - 1)
            process = "\n".join(
                ["".join(lines) for lines in zip(process_header, process_matrix_repr, process_sep, process_noise_repr)]
            )
        else:  # Two lines
            process_header = ["Process: F = "] + ["             "] * (len(process_matrix_repr) - 1)
            process_header += ["", "         Q = "] + ["             "] * (len(process_noise_repr) - 1)
            process = "\n".join(
                ["".join(lines) for lines in zip(process_header, process_matrix_repr + [""] + process_noise_repr)]
            )

        # Measurement string
        with printoptions(profile="short", sci_mode=False, linewidth=100):
            measurement_matrix_repr = str(self.measurement_matrix).split("\n")
            measurement_noise_repr = str(self.measurement_noise).split("\n")

        max_char_matrix = max((len(line) for line in measurement_matrix_repr))
        max_char_noise = max((len(line) for line in measurement_noise_repr))

        if max_char_matrix + max_char_noise <= 110:  # Single line
            measurement_matrix_repr = [line + " " * (max_char_matrix - len(line)) for line in measurement_matrix_repr]
            measurement_header = ["Measurement: H = "] + ["                 "] * (len(measurement_matrix_repr) - 1)
            measurement_sep = ["  &  R = "] + ["         "] * (len(measurement_matrix_repr) - 1)
            measurement = "\n".join(
                [
                    "".join(lines)
                    for lines in zip(
                        measurement_header, measurement_matrix_repr, measurement_sep, measurement_noise_repr
                    )
                ]
            )
        else:  # Two lines
            measurement_header = ["Measurement: H = "] + ["                 "] * (len(measurement_matrix_repr) - 1)
            measurement_header += ["", "             R = "] + ["                 "] * (len(measurement_noise_repr) - 1)
            measurement = "\n".join(
                [
                    "".join(lines)
                    for lines in zip(measurement_header, measurement_matrix_repr + [""] + measurement_noise_repr)
                ]
            )
        n_char = max((len(line) for line in (process + "\n" + measurement).split("\n")))
        return ("\n" + "-" * n_char + "\n").join([header, process, measurement])
