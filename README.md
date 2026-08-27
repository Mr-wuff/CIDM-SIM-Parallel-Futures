# CIDM-SIM Parallel Futures

**Nebius x NVIDIA Global AI Hackathon — Physical AI**

CIDM-SIM Parallel Futures is a stochastic physical-world engine built on the Closed Information Dynamics Model (CIDM/CIDM-BER). It evolves a latent information state and branches multiple plausible futures using a state-conditioned **Stochastic Information Reservoir (SIR)**.

## Core idea

A deterministic long-horizon model can only propagate information already present at initialization. When the observed state is not world-closed, unresolved external information accumulates and point predictors may collapse toward overly smooth conditional means.

CIDM-SIM separates three mechanisms:

1. **CIDM/CIDM-BER** — deterministic information-state and physical evolution.
2. **BER** — minimal open-boundary / energy-closure residual.
3. **SIR** — probabilistic future information innovation.

Conceptually:

```text
Current observations
        |
        v
CIDM information state
        |
        v
CIDM-BER deterministic evolution
        |
        +--------------------------+
        |                          |
        v                          v
Stochastic Information      NVIDIA Nemotron
Reservoir p(xi|psi)         event interpreter
        |                    on Nebius Token Factory
        +------------+-------------+
                     v
             information injection
                     |
          +----------+----------+
          v          v          v
      Future A   Future B   Future C
          |          |          |
          +----------+----------+
                     v
             physics-safe layer
                     |
              UE5 / Web world
```

## Hackathon-period additions

The base CIDM research existed before August 26, 2026. The hackathon build adds substantial new functionality:

- Stochastic Information Reservoir.
- Parallel-future branching and scenario probabilities.
- NVIDIA Nemotron external-event interpretation through Nebius Token Factory.
- Nebius-oriented runtime/deployment integration.
- Browser future-branch viewer.
- UE5 WorldState UDP bridge.
- Optional Tavily runtime retrieval.

## NVIDIA + Nebius role

Nemotron is **not** asked to directly predict the physical future. It converts unstructured external observations into structured candidate event injections. CIDM-SIM remains responsible for physical/information-state evolution.

Configure local credentials through environment variables only:

```text
NEBIUS_API_KEY=
NEBIUS_BASE_URL=
NEBIUS_MODEL=
TAVILY_API_KEY=
```

Never commit real keys to GitHub.

## Quick start

```bash
python -m venv .venv
pip install -r requirements.txt
```

Windows:

```bat
set PYTHONPATH=src
python -m uvicorn app:app --app-dir src --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`.

The branch demo works without cloud credentials. The Nemotron event-interpreter panel requires a real Nebius Token Factory configuration.

## Test

```bash
pytest -q
python scripts/preflight.py
```

## Public repository boundary

This repository is the reproducible public competition implementation. Private datasets, partner data, research-only checkpoints, credentials, and unrelated experiment archives are intentionally excluded.

## Scope

Research simulation prototype only; not a certified safety-critical forecasting service.
