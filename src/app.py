from pathlib import Path
import torch
from fastapi import FastAPI,HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from cidm_architectures import CIDMConfig
from parallel_future_engine import CIDMParallelFutureEngine,ParallelFutureConfig
from demo_physics import make_demo_history
from nebius_nemotron import NebiusNemotronClient

app=FastAPI(title="CIDM-SIM Parallel Futures",version="0.1")
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
CIDM_CFG=CIDMConfig(channels=1,hist_len=4,horizon=12,latent_dim=16,hidden=32,port_channels=1)
PF_CFG=ParallelFutureConfig(branches=5,horizon=12)
ENGINE=CIDMParallelFutureEngine(CIDM_CFG,PF_CFG).to(DEVICE).eval()
NEMOTRON=NebiusNemotronClient()

class EventRequest(BaseModel):
    observation:str

@app.get("/health")
def health():
    return {"ok":True,"device":DEVICE,"nemotron_configured":NEMOTRON.configured,"project":"CIDM-SIM Parallel Futures"}

@app.post("/api/interpret-event")
def interpret_event(req:EventRequest):
    if not NEMOTRON.configured: raise HTTPException(503,"Nemotron/Nebius is not configured.")
    return NEMOTRON.interpret_event(req.observation,{"domain":"physical-world-demo"})

@app.get("/api/demo-branches")
def demo_branches():
    x=torch.from_numpy(make_demo_history()).to(DEVICE)
    out=ENGINE.branch_rollout(x)
    pred=out["pred"].cpu().numpy(); probs=out["branch_probability"].cpu().numpy(); trace=out["innovation_trace"].cpu().numpy()
    branches=[]
    for b in range(pred.shape[0]):
        branches.append({"branch":b,"probability":float(probs[b]),
                         "trajectory":[float(v) for v in pred[b].mean(axis=(1,2,3))],
                         "innovation":[float(v) for v in trace[b]]})
    return {"branches":branches,"device":DEVICE}

@app.get("/",response_class=HTMLResponse)
def home():
    return Path(__file__).resolve().parents[1].joinpath("web/index.html").read_text(encoding="utf-8")
