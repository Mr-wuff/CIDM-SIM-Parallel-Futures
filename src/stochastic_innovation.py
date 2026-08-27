from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class InnovationConfig:
    latent_dim: int = 32
    hidden: int = 96
    event_types: int = 6
    min_scale: float = 1e-3
    max_scale: float = 0.20
    temperature: float = 0.85

class StochasticInformationReservoir(nn.Module):
    """State-conditioned future-information innovation prior.

    BER remains the physical open-boundary/energy closure term.
    SIR represents unresolved future information that may enter the world.
    """
    def __init__(self, cfg: InnovationConfig):
        super().__init__()
        self.cfg = cfg
        D, H = cfg.latent_dim, cfg.hidden
        self.context = nn.Sequential(nn.Conv2d(D,H,3,padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d(1))
        self.mu = nn.Linear(H,D)
        self.log_scale = nn.Linear(H,D)
        self.event_logits = nn.Linear(H,cfg.event_types)
        self.spatial_gate = nn.Sequential(nn.Conv2d(D,H,3,padding=1), nn.SiLU(), nn.Conv2d(H,1,1), nn.Sigmoid())

    def distribution(self, psi):
        c = self.context(psi).flatten(1)
        scale = (F.softplus(self.log_scale(c)) + self.cfg.min_scale).clamp(max=self.cfg.max_scale)
        return {"mu": self.mu(c), "scale": scale, "event_logits": self.event_logits(c)/self.cfg.temperature}

    def sample(self, psi, generator: Optional[torch.Generator]=None):
        d = self.distribution(psi)
        eps = torch.randn(d["mu"].shape, device=psi.device, dtype=psi.dtype, generator=generator)
        z = d["mu"] + d["scale"]*eps
        innovation = self.spatial_gate(psi) * z[:,:,None,None]
        event_prob = torch.softmax(d["event_logits"],dim=-1)
        event_type = torch.multinomial(event_prob,1,generator=generator).squeeze(-1)
        return innovation, {
            "innovation_abs_mean": innovation.abs().mean(),
            "event_prob": event_prob,
            "event_type": event_type,
            "innovation_scale_mean": d["scale"].mean(),
        }

def physics_safe_clip(delta, reference, ratio=0.08):
    ref = reference.detach().abs().mean(dim=(1,2,3),keepdim=True).clamp_min(1e-5)
    lim = ratio*ref
    return delta.clamp(min=-lim,max=lim)
