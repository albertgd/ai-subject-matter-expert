"""
collect_data.py — Alias for research.py (kept for backwards compatibility).

Use research.py instead:
    python scripts/research.py
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).parent / "research.py"
    sys.exit(subprocess.call([sys.executable, str(script)] + sys.argv[1:]))
