#!/usr/bin/env python3
"""Compatibility wrapper for the road marking classifier CLI."""

from road_marking_classifier.cli import main as _cli_main

main = _cli_main


if __name__ == "__main__":
    raise SystemExit(main())
