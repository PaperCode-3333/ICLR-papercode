"""B025 benchmark panel."""
import sys
from pathlib import Path
_RELEASE_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_RELEASE_SRC) not in sys.path:
    sys.path.insert(0, str(_RELEASE_SRC))
