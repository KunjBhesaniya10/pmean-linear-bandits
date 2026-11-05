import os
import sys

# Ensure src is in sys.path so pmlb is importable without requiring pip install -e .
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
