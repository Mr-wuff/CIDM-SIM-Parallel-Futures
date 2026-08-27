# CIDM-SIM Parallel Futures Architecture

## CIDM backbone
Spatial encoder + history encoder + phase/spectral encoder -> latent information state Ψ -> closed dynamics core -> decoder.

## CIDM-BER
Adds:
- SOC: global/low-frequency latent spectral mixing.
- BER: minimal open-system residual closure for boundary input, dissipation, and unresolved exchange.

## Stochastic Information Reservoir (SIR)
SIR is separate from BER:
- BER asks: what minimal residual closes the represented physical system?
- SIR asks: what unresolved future information could plausibly enter next?

## Parallel futures
Each sampled innovation creates a different future branch. Branch probabilities are scenario weights, not deterministic truth claims.

## NVIDIA Nemotron on Nebius
Nemotron interprets unstructured external observations into structured event injections; it does not replace the physical propagator.

## UE5
The UE5 WorldState runtime consumes branch-state streams. The public hackathon repository contains a lightweight UDP bridge; the full UE5 project can remain a separate engine artifact.
