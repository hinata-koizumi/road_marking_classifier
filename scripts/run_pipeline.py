#!/usr/bin/env python3
"""
Road Marking Classifier - Main Pipeline Runner
道路標示分類システム メインパイプライン実行スクリプト
"""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_marking_classifier.cli.main import main

if __name__ == "__main__":
    main()