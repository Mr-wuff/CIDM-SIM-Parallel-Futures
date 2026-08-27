# Hackathon experiment plan

## E0 Runtime integrity
CPU/GPU smoke test; probability normalization; no NaN/Inf; API health.

## E1 Deterministic vs stochastic rollout
Compare CIDM, CIDM-BER, Gaussian-noise baseline, CIDM-BER+SIR.
Metrics: RMSE, MAE, spectral retention, latent energy drift, sharpness, branch diversity.

## E2 Probabilistic calibration
CRPS, Energy Score, empirical coverage, rank histogram, spread-error correlation.

## E3 Innovation ablation
No innovation / fixed Gaussian / state-conditioned Gaussian / event+continuous innovation.

## E4 Nemotron event interpretation
Schema validity, event-type accuracy, probability calibration, location/magnitude consistency, latency and cost.

## E5 UE5
One current world state -> multiple branch trajectories -> live branch switching.
