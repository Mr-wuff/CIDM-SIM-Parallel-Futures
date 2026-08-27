@echo off
set PYTHONPATH=%~dp0..\src
python -m uvicorn app:app --app-dir "%~dp0..\src" --host 127.0.0.1 --port 8000
pause
