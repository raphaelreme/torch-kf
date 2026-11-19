# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-11-19

### Breaking Changes

- **`KalmanFilter.filter`**: the `update_first` and `return_all` flags are now
  keyword-only. Calls that passed them positionally
  (e.g. `kf.filter(state, measures, False, True)`) must be updated to
  `kf.filter(state, measures, update_first=False, return_all=True)`.
- **`KalmanFilter.rts_smooth`**: the `inplace` flag is now keyword-only. Calls
  that passed it positionally (e.g. `kf.rts_smooth(state, True)`) must be
  updated to `kf.rts_smooth(state, inplace=True)`.
- **`KalmanFilter.steady_state_covariance`**: the `n` argument was renamed to
  `n_iter`. Callers passing it by keyword (`steady_state_covariance(n=...)`)
  must switch to `n_iter=...`; positional calls are unaffected.

### Added

- **Time-varying models in `KalmanFilter.filter`**: new optional
  `process_matrix` (`F`), `process_noise` (`Q`), `measurement_matrix` (`H`) and
  `measurement_noise` (`R`) overrides. A leading time axis supplies one matrix
  per step; otherwise the matrix is shared across every step. Measurement
  overrides (`H`, `R`) are indexed per measure (time length `T`), while process
  overrides (`F`, `Q`) are indexed per prediction (time length `T - 1` when
  `update_first`, `T` otherwise).
- **Time-varying models in `KalmanFilter.rts_smooth`**: new optional
  `process_matrix` (`F`) and `process_noise` (`Q`) overrides, following the same
  convention as `filter` (a leading time axis of length `T - 1` supplies one
  matrix per transition, otherwise the matrix is shared across transitions).

### Changed

- **`KalmanFilter.steady_state_covariance`**: the default number of iterations
  (`n_iter`) was increased from `5` to `10` for more reliable convergence on
  slow-converging systems.
- Passing a batched matrix override (more than two dimensions) whose leading
  dimension does not match the expected number of steps now raises a
  `UserWarning` (`"... has no leading time axis of length N; treating it as
  constant over time."`), since this is a likely mistake. Give the override an
  explicit leading time axis of the right length to use it as time-varying (and
  silence the warning).
