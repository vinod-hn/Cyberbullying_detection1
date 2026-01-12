#!/usr/bin/env python
"""
Run script for Cyberbullying Detection API.

Usage:
    python run_api.py
"""

import sys
import os
from pathlib import Path

# Set up paths
project_dir = Path(__file__).parent.absolute()
api_dir = project_dir / "06_api"

# Add paths for imports
sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(api_dir))
sys.path.insert(0, str(api_dir / "routes"))
sys.path.insert(0, str(api_dir / "schemas"))
sys.path.insert(0, str(api_dir / "middleware"))

# Change to project directory
os.chdir(str(project_dir))

# Now import and run
if __name__ == "__main__":
    import uvicorn
    
    # Import settings
    from app_config import settings
    
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"Project directory: {project_dir}")
    print(f"API directory: {api_dir}")
    print(f"Server: http://{settings.host}:{settings.port}")
    print(f"Docs: http://{settings.host}:{settings.port}/docs")
    print("-" * 50)
    
    # Run the server
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        app_dir=str(api_dir)
    )
