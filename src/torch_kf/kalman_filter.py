from __future__ import annotations

import contextlib
import dataclasses
from typing import overload

import torch
import torch.linalg

# Note on runtime:
# Computations may be faster using cholesky decomposition and cholesky_solve
# But in real cases, dim_z is small (limiting the benefits of cholesky)
# Moreover, it is common to compute likelihood (or mahalanobis distance) after an update step which requires an
# additional cholesky solve at each call (even if the decomposition is stored),
# whereas when the inverse is computed, it can be re-used.


if hasattr(torch._tensor_str, "printoptions"):  # noqa: SLF001
    printoptions = torch._tensor_str.printoptions  # noqa: SLF001
else:

    @contextlib.contextmanager
    def printoptions(**kwargs):
        """Change pytorch printoptions temporarily. From the future of pytorch."""
        old_printoptions = torch._tensor_str.PRINT_OPTS  # noqa: SLF001
        torch.set_printoptions(**kwargs)
        try:
            yield
        finally:
            torch._tensor_str.PRINT_OPTS = old_printoptions  # noqa: SLF001


@dataclasses.dataclass
class GaussianState:
    """Gaussian state for Kalman filtering.

    This dataclass stores a multivariate Gaussian distribution:

        x ~ N(mean, covariance)

    Conventions:
    - State/measurement vectors are **column vectors** with shape ``(..., dim, 1)``.
      This avoids ambiguity with batched matrix multiplications.
    - Leading dimensions ``...`` are treated as **batch dimensions** and may be
      broadcastable across operations.

    In addition, an optional precision matrix (inverse covariance) can be stored.
    When present, it can speed up repeated computations such as Mahalanobis distance
    and likelihood evaluation.

    Attributes:
        mean: Mean of the distribution.
            Shape: ``(..., dim, 1)``
        covariance: Covariance matrix of the distribution.
            Shape: ``(..., dim, dim)``
        precision: Optional precision matrix (inverse covariance).
            Shape: ``(..., dim, dim)``
            If ``None``, it may be computed lazily by some methods.
    """

    mean: torch.Tensor
    covariance: torch.Tensor
    precision: torch.Tensor | None = None

    def clone(self) -> GaussianState:
        """Return a deep copy of the state.

        Uses ``Tensor.clone()`` on all stored tensors.

        Returns:
            GaussianState: The cloned state
        """
        return GaussianState(
            self.mean.clone(), self.covariance.clone(), self.precision.clone() if self.precision is not None else None
        )

    def __getitem__(self, idx) -> GaussianState:
        """Index/slice along batch dimensions.

        Notes:
            The index is applied to the leading dimensions of ``mean``, ``covariance``,
            and ``precision`` (if present).

        Args:
            idx (Any): Index/slice applied to the leading batch dimensions.

        Returns:
            GaussianState: Indexed GaussianState.
        """
        return GaussianState(
            self.mean[idx], self.covariance[idx], self.precision[idx] if self.precision is not None else None
        )

    def __setitem__(self, idx, value: GaussianState) -> None:
        """Assign into batch dimensions.

        Args:
            idx (Any): Index/slice applied to the leading batch dimensions to be modified.
            value (GaussianState): GaussianState with compatible shapes.
        """
        if isinstance(value, GaussianState):
            self.mean[idx] = value.mean
            self.covariance[idx] = value.covariance
            if self.precision is not None and value.precision is not None:
                self.precision[idx] = value.precision

            return

        raise NotImplementedError("Only GaussianState assignment is supported.")

    @overload
    def to(self, dtype: torch.dtype) -> GaussianState: ...

    @overload
    def to(self, device: torch.device) -> GaussianState: ...

    def to(self, fmt):
        """Convert a GaussianState to a specific device or dtype.

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
        """Compute squared Mahalanobis distance to a measure.

        Computes:

            MAHA^2 = (x - μ)^T P^{-1} (x - μ)

        Broadcasting:
            Batch dimensions of ``measure`` and the state must be broadcastable.
            This allows to compare multiple states and measures at once.

        Args:
            measure (torch.Tensor): Measure(s) to evaluate (column vector).
                Shape: ``(..., dim, 1)``

        Returns:
            torch.Tensor: Squared Mahalanobis distance for broadcasted measures & states
            Shape: ``(...)``
        """
        diff = self.mean - measure  # Should be broadcastable
        if self.precision is None:
            # Covariance is symmetric and using `.mT` yields a contiguous tensor.
            # This is mathematically equivalent but leads to faster computations though
            # it slightly increases floating point error.
            self.precision = self.covariance.inverse().mT
        return (diff.mT @ self.precision @ diff)[..., 0, 0]

    def mahalanobis(self, measure: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distance to a measure.

        It takes the square root of the squared Mahalanobis distance.

        Notes:
            ``sqrt`` can be relatively expensive. If you only need to compare
            to a threshold, consider comparing squared distances instead.

        Broadcasting:
            Batch dimensions of ``measure`` and the state must be broadcastable.
            This allows to compare multiple states and measures at once.

        Args:
            measure (torch.Tensor): Measure(s) to evaluate (column vector).
                Shape: ``(..., dim, 1)``

        Returns:
            torch.Tensor: Mahalanobis distance for broadcasted measures & states
            Shape: ``(...)``
        """
        return self.mahalanobis_squared(measure).sqrt()

    def log_likelihood(self, measure: torch.Tensor) -> torch.Tensor:
        """Compute the log-likelihood of the given measure under the Gaussian distribution.

        For dimension ``dim``:

            log p(x) = -1/2 * ( dim*log(2π) + log|Σ| + MAHA^2 )

        Broadcasting:
            Batch dimensions of ``measure`` and the state must be broadcastable.
            This allows to compare multiple states and measures at once.

        Args:
            measure (torch.Tensor): Measure(s) to evaluate (column vector).
                Shape: ``(..., dim, 1)``

        Returns:
            torch.Tensor: Log-likelihood for broadcasted measures & states
            Shape: ``(...)``
        """
        maha_2 = self.mahalanobis_squared(measure)
        log_det = torch.log(torch.det(self.covariance))
        pi = torch.tensor(torch.pi, device=log_det.device, dtype=log_det.dtype)
        dim = self.covariance.shape[-1]
        return -0.5 * (dim * torch.log(2 * pi) + log_det + maha_2)

    def likelihood(self, measure: torch.Tensor) -> torch.Tensor:
        """Compute the likelihood of the given measure under the Gaussian distribution.

        It takes the exponential of the log-likelihood.

        Notes:
            ``exp`` can be relatively expensive. If you only need to compare
            likelihoods, consider comparing log-likelihoods instead.

        Broadcasting:
            Batch dimensions of ``measure`` and the state must be broadcastable.
            This allows to compare multiple states and measures at once.

        Args:
            measure (torch.Tensor): Measure(s) to evaluate (column vector).
                Shape: ``(..., dim, 1)``

        Returns:
            torch.Tensor: Likelihood for broadcasted measures & states
            Shape: ``(...)``
        """
        return self.log_likelihood(measure).exp()


class KalmanFilter:
    """Fast and Batch-friendly Kalman filter implementation in PyTorch.

    This class estimates the latent state of a linear dynamical system under Gaussian noise:

        x_k = F x_{k-1} + w_k,   w_k ~ N(0, Q)
        z_k = H x_k     + v_k,   v_k ~ N(0, R)

    where:
    - ``x_k`` is the hidden state (dimension ``dim_x``),
    - ``z_k`` is the measure (dimension ``dim_z``),
    - ``F`` is the transition (process) matrix,
    - ``Q`` is the process noise covariance,
    - ``H`` is the measurement/projection matrix,
    - ``R`` is the measurement noise covariance.

    Kalman filter allows to estimate x_k | z_{1:k} ~ N(mu_k, P_k), i.e. estimate the mean mu_k
    and the covariance P_k of the hidden state, after observing measures for t=1 to t=k.
    Note that in this class, we design by mu_k and P_k both the prior and posterior state mean and covariance.
    Please refer to https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python to understand the differences.

    This class is based on filterpy (https://filterpy.readthedocs.io/en/latest/) numpy implementation of Kalman
    filtering. In contrast, this implementations allows batch processing with cpu and gpu parallelization.

    Shape conventions:
    - Vectors are **column vectors** with shape ``(..., dim, 1)``.
    - Matrices have shape ``(..., dim, dim)`` (or ``(..., dim_z, dim_x)`` for ``H``).
    - Leading ``...`` batch dimensions may be broadcastable, enabling fast batched filtering & smoothing.

    Numerical notes:
    - This implementation is tuned for speed: it typically runs in float32 and uses explicit matrix
    inverses rather than Cholesky-based solves, which are often slower when dim_x/dim_z are small (commonly < 10).
    - For improved numerical robustness, consider running in float64 and enabling joseph_update
    instead of the standard covariance update.

    Attributes:
        process_matrix (torch.Tensor): Process/Transition matrix ``F``.
            Shape: ``(..., dim_x, dim_x)``
        measurement_matrix (torch.Tensor): Projection/Measurement matrix ``H``.
            Shape: ``(..., dim_z, dim_x)``
        process_noise (torch.Tensor): Process noise covariance ``Q``.
            Shape: ``(..., dim_x, dim_x)``
        measurement_noise (torch.Tensor): Measurement noise covariance ``R``.
            Shape: ``(..., dim_z, dim_z)``
        joseph_update (bool): If True, use the Joseph form covariance update for improved numerical stability.
            This is typically ~50% slower than the standard update.
            Default: False
        inv_t (bool): If True, uses ``torch.inv(...).mT`` to inverse, yielding contiguous tensors.
            As covariances are symmetric, this is mathematically equivalent and this should lead to faster computations.
            However, it seems to cause numerical instability, not worth the computational boost.
            Default: False
    """

    _REPR_SPLIT_LENGTH = 110

    def __init__(
        self,
        process_matrix: torch.Tensor,
        measurement_matrix: torch.Tensor,
        process_noise: torch.Tensor,
        measurement_noise: torch.Tensor,
        *,
        joseph_update=False,
        inv_t=False,
    ) -> None:
        # We do not check that any device/dtype/shape are shared (but they should be)
        self.process_matrix = process_matrix
        self.measurement_matrix = measurement_matrix
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self._alpha_sq = 1.0  # Memory fadding KF (As in filterpy)
        self.joseph_update = joseph_update
        self.inv_t = inv_t

    @property
    def state_dim(self) -> int:
        """Dimension of the state variable."""
        return self.process_matrix.shape[-1]

    @property
    def measure_dim(self) -> int:
        """Dimension of the measured variable."""
        return self.measurement_matrix.shape[-2]

    @property
    def device(self) -> torch.device:
        """Device of the Kalman filter."""
        return self.process_matrix.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the Kalman filter."""
        return self.process_matrix.dtype

    @overload
    def to(self, dtype: torch.dtype) -> KalmanFilter: ...

    @overload
    def to(self, device: torch.device) -> KalmanFilter: ...

    def to(self, fmt):
        """Convert a Kalman filter to a specific device or dtype.

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
        process_matrix: torch.Tensor | None = None,
        process_noise: torch.Tensor | None = None,
    ) -> GaussianState:
        """Compute the predicted (prior) state.

        From a state x_{k-1} | ... ~ N(mu_{k-1}, P_{k-1}), it applies the process model:

            x_k = F x_{k-1} + w_k,   w_k ~ N(0, Q)

        leading to a prior state on the next timestep x_k | ... ~ N(mu_k, P_k) with:

            mu_k = F mu_{k-1}
            P_k = F P_{k-1} Fᵀ + Q

        Broadcasting:
            Batch dimensions of ``state``, ``process_matrix`` and ``process_noise`` must be broadcastable.
            It supports multiple states / models as long as it broadcasts correctly.

        Example:
        ```python
            # Initialize a random batch of gaussian state (5d)
            state = GaussianState(
                torch.randn(50, 5, 1),  # The last dimension is required.
                torch.randn(50, 5, 5),
            )

            # Use a single process model
            process_matrix = torch.randn(5, 5)  # Compatible with (50, 5, 5)
            process_noise = torch.randn(5, 5)

            predicted = kf.predict(state, process_matrix=process_matrix, process_noise=process_noise)
            predicted.mean  # Shape: (50, 5, 1)  # Predictions for each state
            predicted.covariance  # Shape: (50, 5, 5)

            # Use several models
            process_matrix = torch.randn(10, 1, 5, 5)  # Compatible with (50, 5, 5)
            process_noise = torch.randn(1, 1, 5, 5)  # Let's use the same noise for each process matrix

            predicted = kf.predict(state, process_matrix=process_matrix, process_noise=process_noise)
            predicted.mean  # Shape: (10, 50, 5, 1)  # Predictions for each model and state
            predicted.covariance  # Shape: (10, 50, 5, 5)
        ```

        Args:
            state: Current posterior state estimate at time k-1.
                Mean shape: ``(..., dim_x, 1)``
                Cov shape:  ``(..., dim_x, dim_x)``
            process_matrix: Optional override for ``F``.
                Shape: ``(..., dim_x, dim_x)``
            process_noise: Optional override for ``Q``.
                Shape: ``(..., dim_x, dim_x)``

        Args:
            state (GaussianState): Current state estimation.
                Shape (mean): ``(..., dim_x, 1)``
                Shape (covariance): ``(..., dim_x, dim_x)``
            process_matrix (torch.Tensor | None): Optional override for registered process matrix ``F``.
                Shape: ``(..., dim_x, dim_x)``
            process_noise (torch.Tensor | None): Optional override for registered process noise ``Q``.
                Shape: ``(..., dim_x, dim_x)``

        Returns:
            GaussianState: Predicted prior state on the next time frame.
                Shape (mean): ``(..., dim_x, 1)``
                Shape (covariance): ``(..., dim_x, dim_x)``

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
        measurement_matrix: torch.Tensor | None = None,
        measurement_noise: torch.Tensor | None = None,
        precompute_precision=True,
    ) -> GaussianState:
        r"""Project a state into measurement space (usually the predicted state).

        From a state x_k | ... ~ N(mu_k, P_k), it applies the measurement model:

            z_k = H x_k + v_k,   v_k ~ N(0, R)

        leading to a Gaussian state over ``z``: z_k | ... ~ N(y_k, S_k) with:

            y_k = H mu_k
            S_k = H P_k Hᵀ + R

        Broadcasting:
            Batch dimensions of ``state``, ``measurement_matrix`` and ``measurement_noise`` must be broadcastable.
            It supports multiple states / models as long as it broadcasts correctly.

        Example:
        ```python
            # Initialize a random batch of gaussian state (5d)
            state = GaussianState(
                torch.randn(50, 5, 1),  # The last dimension is required.
                torch.randn(50, 5, 5),
            )

            # Use a single projection model
            measurement_matrix = torch.randn(3, 5)  # Compatible with (50, 5, 5)
            measurement_noise = torch.randn(3, 3)  # Broadcastable with (50, 3, 3)

            projection = kf.project(state, measurement_matrix=measurement_matrix, measurement_noise=measurement_noise)
            projection.mean  # Shape: (50, 3, 1)  # projection for each state
            projection.covariance  # Shape: (50, 3, 3)

            # Use several models
            measurement_matrix = torch.randn(1, 1, 3, 5)  # Same measurement for each model, compatible with (50, 5, 5)
            measurement_noise = torch.randn(10, 1, 3, 3)  # Use different noises

            projection = kf.project(state, measurement_matrix=measurement_matrix, measurement_noise=measurement_noise)
            projection.mean  # Shape: (1, 50, 3, 1)  # WARNING: the state will not be broadcasted to (10, 50, 5, 1).
            projection.covariance  # Shape: (10, 50, 3, 3)  # Projection cov for each model and each state
        ```

        Args:
            state (GaussianState): Current state estimation, typically the results of `predict`.
                Shape (mean): ``(..., dim_x, 1)``
                Shape (covariance): ``(..., dim_x, dim_x)``
            measurement_matrix (torch.Tensor | None): Optional override for registered projection matrix ``H``.
                Shape: ``(..., dim_z, dim_x)``
            measurement_noise (torch.Tensor | None): Optional override for registered projection noise ``R``.
                Shape: ``(..., dim_z, dim_z)``
            precompute_precision (bool): If True, compute and store ``S^{-1}`` in the returned state's ``precision``.
                Useful for `update` or evaluate likelihoods, avoiding several inverse computations.
                Default: True

        Returns:
            GaussianState: Projected state in the measurement space.
                Shape (mean): ``(..., dim_z, 1)``
                Shape (covariance): ``(..., dim_z, dim_z)``

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
                # The inverse can be transposed (back) to be contiguous: as it is symmetric
                # This is equivalent and faster to hold on the contiguous version
                # But this may slightly increase floating errors.
                (covariance.inverse().mT if self.inv_t else covariance.inverse()) if precompute_precision else None
            ),
        )

    def update(
        self,
        state: GaussianState,
        measure: torch.Tensor,
        *,
        projection: GaussianState | None = None,
        measurement_matrix: torch.Tensor | None = None,
        measurement_noise: torch.Tensor | None = None,
    ) -> GaussianState:
        """Update a state estimate using a new measure.

        Given a state x_k | ... ~ N(mu_k, P_k) and a new observation z_k. It
        computes the posterior state x_k | ..., z_k ~ N(mu'_k, P'_k), accounting
        for the new measure z_k.

        `update` follows three main steps:
        1. Computing the measure expected distribution z_k | ... ~ N(y_k, S_k) with `project`.
        2. Kalman gain computation: K = P_k Hᵀ S_k^{-1}
        3. Incorporate z_k information in the state:
            mu'_k = mu_k + K (z_k - y_k)
            P'_k = (I - K H) P_k   OR [JOSEPH_UPDATE] P'_k = (I - K H) P_k (I - K H)ᵀ + K R Kᵀ

        Broadcasting:
            Batch dimensions of ``state``, `measure``, ``projection``, ``measurement_matrix`` and ``measurement_noise``
            must be broadcastable.
            It supports multiple states, measures and models as long as it broadcasts correctly.

        Example:
        ```python
            # Initialize a random batch of gaussian state (5d)
            state = GaussianState(
                torch.randn(50, 5, 1),  # The last dimension is required.
                torch.randn(50, 5, 5),
            )

            # Use a single projection model and a single measurement for each state
            measurement_matrix = torch.randn(3, 5)  # Compatible with (50, 5, 5)
            measurement_noise = torch.randn(3, 3)  # Broadcastable with (50, 3, 3)
            measure = torch.randn(50, 3, 3)

            new_state = kf.update(
                state, measure, measurement_matrix=measurement_matrix, measurement_noise=measurement_noise
            )
            new_state.mean  # Shape: (50, 5, 1)  # Each state has been updated
            new_state.covariance  # Shape: (50, 5, 5)

            # Use several models and a single measurement for each state
            measurement_matrix = torch.randn(10, 1, 3, 5)  # Compatible with (50, 5, 5)
            measurement_noise = torch.randn(10, 1, 3, 3)  # Use different noises
            measure = torch.randn(50, 3, 1)  # The last unsqueezed dimension is required

            new_state = kf.update(
                state, measure, measurement_matrix=measurement_matrix, measurement_noise=measurement_noise
            )
            new_state.mean  # Shape: (10, 50, 5, 1)  # Each state for each model has been updated
            new_state.covariance  # Shape: (10, 50, 5, 1)

            # Use several models and all measurements for each state
            measurement_matrix = torch.randn(10, 1, 3, 5)  # Compatible with (50, 5, 5)
            measurement_noise = torch.randn(10, 1, 3, 3)  # Use different noises
            # We have 50 measurements and we update each state/model with every measurements
            measure = torch.randn(50, 1, 1, 3, 1)

            new_state = kf.update(
                state, measure, measurement_matrix=measurement_matrix, measurement_noise=measurement_noise
            )
            new_state.mean  # Shape: (50, 10, 50, 5, 1)  # Update for each measure, model and previous state
            new_state.covariance  # Shape: (10, 50, 5, 5)  # WARNING: The cov is not broadcasted to (50, 10, 50, 5, 5)
        ```

        Args:
            state (GaussianState): Current state estimation, typically the results of `predict`.
                Shape (mean): ``(..., dim_x, 1)``
                Shape (covariance): ``(..., dim_x, dim_x)``
            measure (torch.Tensor): Measure of the state `z_k` (column vector).
                Shape: ``(..., dim_z, 1)``
            projection (GaussianState | None): Optional precomputed projection from `project`.
            measurement_matrix (torch.Tensor | None): Optional override for registered projection matrix ``H``.
                Shape: ``(..., dim_z, dim_x)``
            measurement_noise (torch.Tensor | None): Optional override for registered projection noise ``R``.
                Shape: ``(..., dim_z, dim_z)``

        Returns:
            GaussianState: Updated posterior state.
                Shape (mean): ``(..., dim_x, 1)``
                Shape (covariance): ``(..., dim_x, dim_x)``

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
            chol_decomposition, _ = torch.linalg.cholesky_ex(projection.covariance)
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
        self, state: GaussianState, measures: torch.Tensor, update_first=True, return_all=False
    ) -> GaussianState:
        """Run the classic predict/update loop over a sequence of measures.

        This is a convenience method for common use cases. It assumes:
        - A fixed model over time (constant ``F, Q, H, R``).
        - Measurements are already aligned with the batch of states.
        - Measurements may contain NaNs: if any component of a measurement vector is NaN,
          the corresponding state is **not** updated at that timestep.

        For more complex filtering approaches, one should directly implements filtering with
        dedicated `predict` and `update` calls. (e.g., supporting time varying models; aligning
        measures with states).

        Args:
            state (GaussianState): Initial prior on the state at t=0, before seeing any of the measures.
                Shape (mean): ``(..., dim_x, 1)``
                Shape (covariance): ``(..., dim_x, dim_x)``
            measures (torch.Tensor): Sequence of measures over time.
                Shape: ``(T, ..., dim_z, 1)``
            update_first (bool): If True, skip the prediction step on the first timestep, such that the initial state
                corresponds to the prior at t=0.
                Default: True
            return_all (bool): If True, return the posterior state at every timestep as a single `GaussianState`
                with a leading time dimension. The prior states can be accessed with an additional `predict`.
                If False, it only returns the last posterior state.
                Default: False

        Returns:
            GaussianState: Either the last posterior state, or all the posterior states.
                Shape (mean): ``([T, ]..., dim_x, 1)``
                Shape (covariance): ``([T, ]..., dim_x, dim_x)``
        """
        # Convert state to the right dtype and device
        state = state.to(self.dtype).to(self.device)

        saver: GaussianState

        for t, measure in enumerate(measures):
            if t or not update_first:  # Do not predict on the first t
                state = self.predict(state)

            # Convert on the fly the measure to avoid to store them all in cuda memory
            # To avoid this overhead, the conversion can be done by the user before calling `filter`
            measure = measure.to(self.dtype).to(self.device, non_blocking=True)  # noqa: PLW2901

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
        """Apply Rauch-Tung-Striebel (RTS) smoothing to filtered states.

        Input is assumed to be the sequence of **filtered (posterior) states** over time,
        typically obtained with `filter(return_all=True)`.

        Notes:
            - Uses a fixed model over time (constant ``F, Q``).
            - The time dimension is assumed to be the first dimension of ``state.mean`` and ``state.covariance``.

        More complex smoothing approaches could be implemented using this function as a baseline.

        Args:
            state (GaussianState): Filtered states over time.
                Shape (mean): ``(T, ..., dim_x, 1)``
                Shape (covariance): ``(T, ..., dim_x, dim_x)``
            inplace (bool): If True, modify and return ``state`` in place.
                If False, operate on a cloned copy.
                Default: False

        Returns:
            Smoothed states over time with the same shapes as the input.

        Returns:
            GaussianState: All smoothed states in time, with the same shapes as the input.
                Shape (mean): ``([T, ]..., dim_x, 1)``
                Shape (covariance): ``([T, ]..., dim_x, dim_x)``
        """
        out = state if inplace else GaussianState(state.mean.clone(), state.covariance.clone())

        # Iterate backward to update all states (except the last one which is already fine)
        for t in range(state.mean.shape[0] - 2, -1, -1):
            cov_at_process = state.covariance[t] @ self.process_matrix.mT
            predicted_covariance = self.process_matrix @ cov_at_process + self.process_noise

            kalman_gain = cov_at_process @ (
                predicted_covariance.inverse().mT if self.inv_t else predicted_covariance.inverse()
            )
            out.mean[t] += kalman_gain @ (out.mean[t + 1] - self.process_matrix @ state.mean[t])
            out.covariance[t] += kalman_gain @ (out.covariance[t + 1] - predicted_covariance) @ kalman_gain.mT

        return out

    def __repr__(self) -> str:
        """Convert the Kalman filter model into a readable string."""
        header = f"Kalman Filter (State dimension: {self.state_dim}, Measure dimension: {self.measure_dim})"

        # Process string
        with printoptions(profile="short", sci_mode=False, linewidth=80):
            process_matrix_repr = str(self.process_matrix).split("\n")
            process_noise_repr = str(self.process_noise).split("\n")

        max_char_matrix = max(len(line) for line in process_matrix_repr)
        max_char_noise = max(len(line) for line in process_noise_repr)
        if max_char_matrix + max_char_noise <= self._REPR_SPLIT_LENGTH:  # Single line
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
                ["".join(lines) for lines in zip(process_header, [*process_matrix_repr, "", *process_noise_repr])]
            )

        # Measurement string
        with printoptions(profile="short", sci_mode=False, linewidth=100):
            measurement_matrix_repr = str(self.measurement_matrix).split("\n")
            measurement_noise_repr = str(self.measurement_noise).split("\n")

        max_char_matrix = max(len(line) for line in measurement_matrix_repr)
        max_char_noise = max(len(line) for line in measurement_noise_repr)

        if max_char_matrix + max_char_noise <= self._REPR_SPLIT_LENGTH:  # Single line
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
                    for lines in zip(measurement_header, [*measurement_matrix_repr, "", *measurement_noise_repr])
                ]
            )
        n_char = max(len(line) for line in (process + "\n" + measurement).split("\n"))
        return ("\n" + "-" * n_char + "\n").join([header, process, measurement])
