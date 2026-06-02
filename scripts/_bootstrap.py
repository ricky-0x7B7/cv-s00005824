"""Rende la radice del repository importabile in modo che `from src...` funzioni quando si
eseguono gli script direttamente (``python scripts/foo.py``) senza installare il pacchetto."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
