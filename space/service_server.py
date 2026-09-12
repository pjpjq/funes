#!/usr/bin/env python3
"""HF Space entry point for the durable Python compatibility service.

The native Funes bridge remains in ``space/server.py`` for local/native
experiments.  This entry point is used for the large historical backfill: the
Hub dataset stores compressed raw/delta snapshots, while SQLite FTS provides a
fast restart-safe retrieval surface.  Native Lance indexes remain rebuildable
from the same raw source.
"""
from service.server import serve


if __name__ == "__main__":
    serve("0.0.0.0", int(__import__("os").environ.get("PORT", "7860")))
