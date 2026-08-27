from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from cidm_architectures import CIDMConfig, CIDM_BER
from stochastic_innovation import InnovationConfig, StochasticInformationReservoir, physics_safe_clip

@dataclass
class ParallelFutureConfig:
    branches: int = 6
    horizon: int = 12
    innovation_every: int = 1
    innovation_strength: float = 0.35
    physics_clip_ratio: float = 0.08

class CIDMParallelFutureEngine(nn.Module):
    def __init__(self, cidm_cfg: CIDMConfig, pf_cfg: ParallelFutureConfig):
        super().__init__()
        self.cidm = CIDM_BER(cidm_cfg)
        self.sir = StochasticInformationReservoir(InnovationConfig(latent_dim=cidm_cfg.latent_dim))
        self.pf_cfg = pf_cfg

    @torch.no_grad()
    def branch_rollout(self, history, port: Optional[torch.Tensor]=None):
        if history.shape[0] != 1:
            raise ValueError("Demo branch_rollout expects batch size 1.")
        psi0 = self.cidm.encode(history)
        branch_states = psi0.repeat(self.pf_cfg.branches,1,1,1)
        preds, innovation_trace = [], []
        logw = torch.zeros(self.pf_cfg.branches,device=history.device)

        for t in range(self.pf_cfg.horizon):
            new_states, step_preds, step_innov = [], [], []
            for b in range(self.pf_cfg.branches):
                psi = branch_states[b:b+1]
                pstep = None if port is None else (port[:,min(t,port.shape[1]-1)] if port.dim()==5 else port)
                psi_det,_ = self.cidm.step(psi,t,pstep)
                if t % self.pf_cfg.innovation_every == 0:
                    xi,info = self.sir.sample(psi)
                    xi = physics_safe_clip(xi,psi_det,self.pf_cfg.physics_clip_ratio)
                    psi_next = psi_det + self.pf_cfg.innovation_strength*xi
                    logw[b] += -0.5*float(info["innovation_scale_mean"])
                    step_innov.append(float(info["innovation_abs_mean"]))
                else:
                    psi_next = psi_det
                    step_innov.append(0.0)
                new_states.append(psi_next)
                step_preds.append(self.cidm.decode(psi_next))
            branch_states = torch.cat(new_states,dim=0)
            preds.append(torch.cat(step_preds,dim=0))
            innovation_trace.append(step_innov)

        return {
            "pred": torch.stack(preds,dim=1),
            "branch_probability": torch.softmax(logw,dim=0),
            "innovation_trace": torch.tensor(innovation_trace,device=history.device).T,
        }
