"""Public hackathon reference implementation of the CIDM/CIDM-BER family.

This module preserves the competition-relevant structure of the archived CIDM design:
spatial/history/spectral encoding, closed latent dynamics, a spectral operator core (SOC),
and a Boundary Energy Reservoir (BER). Research-only checkpoints and internal experiment
utilities are intentionally excluded from the public hackathon repository.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class CIDMConfig:
    channels:int=1
    hist_len:int=4
    horizon:int=6
    latent_dim:int=32
    hidden:int=64
    port_channels:int=1
    use_phase_encoder:bool=True
    use_time_encoder:bool=True
    use_hidden_reservoir:bool=True
    use_flux:bool=True
    soc_modes1:int=8
    soc_modes2:int=8
    ber_add_scale:float=0.05
    ber_enabled:bool=True
    soc_enabled:bool=True

def _block(cin,cout,h):
    return nn.Sequential(nn.Conv2d(cin,h,3,padding=1),nn.GroupNorm(1,h),nn.SiLU(),nn.Conv2d(h,cout,3,padding=1))

class PhaseSpectralEncoder(nn.Module):
    def __init__(self,c,d):
        super().__init__(); self.proj=nn.Sequential(nn.Conv2d(c*3,d,1),nn.SiLU(),nn.Conv2d(d,d,1))
    def forward(self,x):
        z=torch.fft.fft2(x,norm='ortho'); a=torch.log1p(torch.abs(z)); f=self.proj(torch.cat([z.real,z.imag,a],1))
        return torch.fft.ifft2(torch.complex(f,torch.zeros_like(f)),norm='ortho').real

class ClosedInformationDynamicsCore(nn.Module):
    def __init__(self,d,h):
        super().__init__(); self.local=_block(d,d,h); self.raw=nn.Parameter(torch.randn(d,d)*0.02); self.alpha=nn.Parameter(torch.tensor(0.18))
    def forward(self,psi):
        A=self.raw-self.raw.t(); skew=torch.einsum('ij,bjhw->bihw',A,psi); dpsi=skew+0.1*self.local(psi)
        return psi+torch.tanh(self.alpha)*dpsi

class LatentSpectralOperatorCore(nn.Module):
    def __init__(self,d,h):
        super().__init__(); self.mix=nn.Sequential(nn.Conv2d(d,h,1),nn.SiLU(),nn.Conv2d(h,d,1)); self.scale=nn.Parameter(torch.tensor(-1.8))
    def forward(self,psi):
        z=torch.fft.fft2(psi,norm='ortho'); H,W=psi.shape[-2:]; ky=torch.fft.fftfreq(H,device=psi.device).view(1,1,H,1); kx=torch.fft.fftfreq(W,device=psi.device).view(1,1,1,W)
        filt=torch.exp(-6*(kx*kx+ky*ky)); low=torch.fft.ifft2(z*filt,norm='ortho').real
        return torch.sigmoid(self.scale)*self.mix(low)

class BoundaryEnergyReservoir(nn.Module):
    def __init__(self,d,port,h):
        super().__init__(); self.port=max(1,port); self.net=_block(d*2+self.port,d,h); self.mask=nn.Sequential(nn.Conv2d(d*2+self.port,max(8,h//2),3,padding=1),nn.SiLU(),nn.Conv2d(max(8,h//2),1,1),nn.Sigmoid()); self.scale=nn.Parameter(torch.tensor(-3.0))
    def forward(self,psi,internal,port=None):
        B,_,H,W=psi.shape
        if port is None: p=torch.zeros(B,self.port,H,W,device=psi.device,dtype=psi.dtype)
        else:
            p=port
            if p.shape[-2:]!=(H,W): p=F.interpolate(p,size=(H,W),mode='bilinear',align_corners=False)
            if p.shape[1]<self.port: p=torch.cat([p,torch.zeros(B,self.port-p.shape[1],H,W,device=psi.device,dtype=psi.dtype)],1)
            p=p[:,:self.port]
        x=torch.cat([psi,internal,p],1); raw=torch.tanh(self.net(x)); m=self.mask(x); delta=torch.sigmoid(self.scale)*m*raw
        return delta,{"ber_strength":delta.abs().mean(),"ber_mask_mean":m.mean()}

class CIDM(nn.Module):
    def __init__(self,cfg:CIDMConfig):
        super().__init__(); self.config=cfg; D,H=cfg.latent_dim,cfg.hidden
        self.space=_block(cfg.channels,D,H); self.hist=_block(cfg.channels*cfg.hist_len,D,H); self.phase=PhaseSpectralEncoder(cfg.channels,D)
        n=1+int(cfg.use_time_encoder)+int(cfg.use_phase_encoder); self.fuse=nn.Sequential(nn.Conv2d(D*n,D,1),nn.SiLU(),nn.Conv2d(D,D,1))
        self.core=ClosedInformationDynamicsCore(D,H); self.dec=_block(D,cfg.channels,H)
    def encode(self,history):
        B,L,C,H,W=history.shape; fs=[self.space(history[:,-1])]
        if self.config.use_time_encoder: fs.append(self.hist(history.reshape(B,L*C,H,W)))
        if self.config.use_phase_encoder: fs.append(self.phase(history[:,-1]))
        return self.fuse(torch.cat(fs,1))
    def decode(self,psi): return self.dec(psi)
    def step(self,psi,step_index=0,port=None):
        nxt=self.core(psi); z=torch.zeros((),device=psi.device); return nxt,{"soc_strength":z,"ber_strength":z}

class CIDM_BER(CIDM):
    def __init__(self,cfg:CIDMConfig):
        super().__init__(cfg); self.soc=LatentSpectralOperatorCore(cfg.latent_dim,cfg.hidden); self.ber=BoundaryEnergyReservoir(cfg.latent_dim,cfg.port_channels,cfg.hidden)
    def step(self,psi,step_index=0,port=None):
        psi_int=self.core(psi); soc=self.soc(psi) if self.config.soc_enabled else torch.zeros_like(psi); psi_int=psi_int+soc; internal=psi_int-psi
        if self.config.ber_enabled:
            ber,info=self.ber(psi,internal.detach(),port); nxt=psi_int+self.config.ber_add_scale*ber
        else:
            info={"ber_strength":torch.zeros((),device=psi.device)}; nxt=psi_int
        return nxt,{"soc_strength":soc.abs().mean(),"ber_strength":info["ber_strength"]}

def build_cidm(config:Optional[CIDMConfig]=None): return CIDM(config or CIDMConfig())
def build_cidm_ber(config:Optional[CIDMConfig]=None): return CIDM_BER(config or CIDMConfig())
