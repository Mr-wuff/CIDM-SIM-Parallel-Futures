import json, os, requests

DEFAULT_SYSTEM = """You are the event-interpretation layer for a physical-world simulation.
Convert an unstructured observation into conservative structured candidate external-information injection.
Return JSON only with: event_type, probability, location:[x,y], magnitude, duration, confidence, description.
Do not claim certainty."""

class NebiusNemotronClient:
    def __init__(self):
        self.api_key=os.getenv("NEBIUS_API_KEY","")
        self.base_url=os.getenv("NEBIUS_BASE_URL","").rstrip("/")
        self.model=os.getenv("NEBIUS_MODEL","")
        self.timeout=float(os.getenv("NEBIUS_TIMEOUT","60"))

    @property
    def configured(self):
        return bool(self.api_key and self.base_url and self.model)

    def interpret_event(self, observation, world_context=None):
        if not self.configured:
            raise RuntimeError("Nebius/Nemotron is not configured. See .env.example.")
        payload={
            "model":self.model,
            "messages":[
                {"role":"system","content":DEFAULT_SYSTEM},
                {"role":"user","content":json.dumps({"observation":observation,"world_context":world_context or {}},ensure_ascii=False)}
            ],
            "temperature":0.4
        }
        r=requests.post(f"{self.base_url}/chat/completions",
            headers={"Authorization":f"Bearer {self.api_key}","Content-Type":"application/json"},
            json=payload,timeout=self.timeout)
        r.raise_for_status()
        content=r.json()["choices"][0]["message"]["content"]
        if isinstance(content,dict): return content
        content=content.strip()
        if content.startswith("```"):
            content=content.split("\n",1)[1].rsplit("```",1)[0]
        return json.loads(content)
