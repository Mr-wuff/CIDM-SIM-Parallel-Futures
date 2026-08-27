from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json

@dataclass
class EventInjection:
    event_type: str
    probability: float
    location: List[float] = field(default_factory=lambda: [0.5, 0.5])
    magnitude: float = 0.0
    duration: float = 1.0
    confidence: float = 0.5
    source: str = "innovation_prior"
    description: str = ""

@dataclass
class FutureBranch:
    branch_id: str
    probability: float
    parent_id: Optional[str]
    events: List[EventInjection] = field(default_factory=list)
    summary: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    world_state: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorldEnvelope:
    protocol: str = "CIDM-Parallel-Futures-0.1"
    sequence: int = 0
    sim_time: float = 0.0
    branches: List[FutureBranch] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)
