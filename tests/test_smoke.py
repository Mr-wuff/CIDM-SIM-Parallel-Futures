import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import torch
from cidm_architectures import CIDMConfig
from parallel_future_engine import CIDMParallelFutureEngine,ParallelFutureConfig
from demo_physics import make_demo_history

def test_parallel_future_shapes():
    cfg=CIDMConfig(channels=1,hist_len=4,horizon=3,latent_dim=8,hidden=16)
    model=CIDMParallelFutureEngine(cfg,ParallelFutureConfig(branches=3,horizon=3)).eval()
    x=torch.from_numpy(make_demo_history(size=16))
    out=model.branch_rollout(x)
    assert out["pred"].shape[:2]==(3,3)
    assert torch.isclose(out["branch_probability"].sum(),torch.tensor(1.0),atol=1e-5)
