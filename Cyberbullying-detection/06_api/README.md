# Cyberbullying Detection API

Quick run and test instructions for the modular FastAPI service.

## Prerequisites
- Python 3.11+ (tested with 3.13)
- Windows PowerShell (commands below) or a shell of your choice

## Setup
```powershell
# From repo root
cd Cyberbullying-detection

# Create / activate venv (re-uses parent .venv if present)
if (Test-Path "..\.venv\Scripts\Activate.ps1") { ..\.venv\Scripts\Activate.ps1 } else { py -3 -m venv ..\.venv; ..\.venv\Scripts\Activate.ps1 }

# Install API requirements
pip install -r 06_api\requirements_api.txt
```

## Run
```powershell
# Start on port 8000 (use 8001 if 8000 is busy)
uvicorn 06_api.main:app --host 127.0.0.1 --port 8000
# alt
uvicorn 06_api.main:app --host 127.0.0.1 --port 8001
```

- Docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## Tests
```powershell
# From repo root with venv activated
pytest 06_api\tests -q
```

## Notes
- The package folder is named `06_api`. For Python code that needs to import from it, prefer runtime imports:
  ```python
  import importlib
  main = importlib.import_module("06_api.main")
  app = main.app
  ```
  The test suite follows this pattern.
- Logs are written to `17_logs/api.log` when the logging middleware is enabled.
- Configuration is in `06_api/app_config.py`.
