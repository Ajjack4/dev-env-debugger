"""
OpenEnv server entry point — required for uv run server / openenv validate.
Delegates to the root api.py FastAPI application.
"""

import os
import sys

# Allow imports from the project root (api.py, environment.py, etc.)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import uvicorn
from api import app  # noqa: E402


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
