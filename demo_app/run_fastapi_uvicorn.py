#!/usr/bin/env python3
"""Run Uvicorn with WebSocket keepalive disabled.

During streaming, keepalive pings can race with the WebSocket drain operation and trigger
an AssertionError. To avoid this, we set ws_ping_interval=None (and ws_ping_timeout=None).

The Uvicorn CLI cannot accept None for these options, so we launch Uvicorn from Python.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the CosmosReason2 FastAPI app via Uvicorn")
    parser.add_argument(
        "--host",
        default=os.environ.get("UVICORN_HOST", "0.0.0.0"),
        help="Bind host (default: env UVICORN_HOST or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("UVICORN_PORT", "8001")),
        help="Bind port (default: env UVICORN_PORT or 8001)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.environ.get("UVICORN_RELOAD", "").strip().lower() in {"1", "true", "yes", "on"},
        help="Enable auto-reload (default: env UVICORN_RELOAD)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("UVICORN_LOG_LEVEL", "info"),
        help="Uvicorn log level (default: env UVICORN_LOG_LEVEL or info)",
    )
    args = parser.parse_args()

    uvicorn.run(
        "fastapi_app:app",
        host=args.host,
        port=args.port,
        ws_ping_interval=None,
        ws_ping_timeout=None,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
