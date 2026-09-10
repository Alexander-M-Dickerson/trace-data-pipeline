"""pytest bootstrap: make scripts/ importable (_stage2_settings, lib.*) from the tests dir."""
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
