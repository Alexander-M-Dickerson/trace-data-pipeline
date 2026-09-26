"""pytest bootstrap: make the stage directory importable (_stage4_settings, factorlib) from the tests dir."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
