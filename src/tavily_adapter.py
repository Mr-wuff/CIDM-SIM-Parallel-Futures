import os, requests
class TavilyRuntime:
    def __init__(self): self.api_key=os.getenv("TAVILY_API_KEY","")
    @property
    def configured(self): return bool(self.api_key)
    def search(self,query,max_results=5):
        if not self.configured: raise RuntimeError("TAVILY_API_KEY is not configured.")
        r=requests.post("https://api.tavily.com/search",
            json={"api_key":self.api_key,"query":query,"search_depth":"advanced","max_results":max_results},
            timeout=45)
        r.raise_for_status()
        return r.json()
