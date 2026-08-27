# Devpost Submission Draft

## Project
CIDM-SIM Parallel Futures

## Tagline
A stochastic physical-world engine that branches plausible futures from incomplete information and renders them as interactive scenarios.

## Track
Physical AI

## Inspiration
Real-world prediction is an open-system problem. A model rarely observes a world-closed initial information state, yet deterministic rollout assumes the future is fully implied by what is already known. As horizon grows, unresolved external information increases and point predictions can drift toward smooth conditional averages.

## What it does
CIDM-SIM evolves a latent information state with CIDM/CIDM-BER and generates multiple future branches through a state-conditioned stochastic information-innovation model. NVIDIA Nemotron on Nebius Token Factory converts unstructured event observations into structured candidate information injections. The engine propagates and compares those candidates under CIDM dynamics and physical constraints.

## How we built it
- PyTorch CIDM / CIDM-BER
- SOC for spectral/global latent transport
- BER for open-boundary residual closure
- SIR for probabilistic future information
- Nebius Token Factory + NVIDIA Nemotron
- FastAPI demo/API
- UE5 WorldState bridge
- Optional Tavily runtime retrieval

## What's next
Train posterior/prior SIR pairs on historical trajectories, add CRPS/Energy Score/coverage diagnostics, expand to ocean-atmosphere and embodied-agent worlds, and run scalable ensembles with Nebius Serverless Jobs.
