import os,sys,importlib.util
checks={"python>=3.10":sys.version_info>=(3,10),"torch":importlib.util.find_spec("torch") is not None,"fastapi":importlib.util.find_spec("fastapi") is not None,"NEBIUS_API_KEY":bool(os.getenv("NEBIUS_API_KEY")),"NEBIUS_BASE_URL":bool(os.getenv("NEBIUS_BASE_URL")),"NEBIUS_MODEL":bool(os.getenv("NEBIUS_MODEL"))}
for k,v in checks.items(): print(f"{k:20s} {'OK' if v else 'MISSING'}")
