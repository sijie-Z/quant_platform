#!/usr/bin/env python
"""Launch script for lab runs — avoids PYTHONPATH / -m / -c cross-shell issues."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "quant_platform"))  # flat-layout fallback
# Actually the flat layout means D:/Desktop is the import root
sys.path.insert(0, "D:/Desktop")
from quant_platform.lab.runs.first_honest_research_run import run
run()
