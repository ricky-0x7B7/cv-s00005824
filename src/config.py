"""Configurazione globale del progetto: path, selezione del device, riproducibilità, classi.

Importabile da notebook e script come::

    from src.config import select_device, set_reproducibility, SH17_CLASSES
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Radice del repo = genitore del genitore di questo file (src/ -> radice del repo).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Classi SH17 (ordine autorevole preso dal file ufficiale sh17.yaml).
# Cambiare l'ordine disallineerebbe ogni file di label, quindi va mantenuto congelato.
# --------------------------------------------------------------------------- #
SH17_CLASSES: list[str] = [
    "person",       # 0
    "ear",          # 1
    "ear-mufs",     # 2
    "face",         # 3
    "face-guard",   # 4
    "face-mask",    # 5
    "foot",         # 6
    "tool",         # 7
    "glasses",      # 8
    "gloves",       # 9
    "helmet",       # 10
    "hands",        # 11
    "head",         # 12
    "medical-suit",  # 13
    "shoes",        # 14
    "safety-suit",  # 15
    "safety-vest",  # 16
]

CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(SH17_CLASSES)}

# Classi di DPI rilevanti per il motore di compliance.
PERSON_CLASS = CLASS_TO_ID["person"]
# DPI considerati "essenziali" per la regola SAFE/UNSAFE.
ESSENTIAL_PPE: dict[str, int] = {
    "helmet": CLASS_TO_ID["helmet"],
    "safety-vest": CLASS_TO_ID["safety-vest"],
}
# DPI aggiuntivi tracciati per un reporting più ricco (non richiesti per SAFE).
OPTIONAL_PPE: dict[str, int] = {
    "gloves": CLASS_TO_ID["gloves"],
    "glasses": CLASS_TO_ID["glasses"],
    "shoes": CLASS_TO_ID["shoes"],
    "ear-mufs": CLASS_TO_ID["ear-mufs"],
    "face-mask": CLASS_TO_ID["face-mask"],
}


@dataclass(frozen=True)
class ProjectConfig:
    """Impostazioni statiche, valide per l'intero progetto."""

    seed: int = 42
    image_size: int = 640
    default_device: str = "auto"
    data_root: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    processed_root: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "sh17")
    models_root: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    reports_root: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")
    outputs_root: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")


CONFIG = ProjectConfig()


def select_device(preferred: str = "auto") -> str:
    """Restituisce la migliore stringa di device torch disponibile.

    Parameters
    ----------
    preferred:
        Uno tra ``"auto"``, ``"mps"``, ``"cuda"``, ``"cpu"``. Con ``"auto"`` l'ordine
        di preferenza è MPS (Apple Silicon) -> CUDA -> CPU.

    Returns
    -------
    str
        Una stringa di device compatibile con PyTorch e Ultralytics.
    """
    if preferred != "auto":
        return preferred

    # Importa torch in modo lazy, così i percorsi di codice non-DL (es. l'audit del
    # dataset) non pagano il costo dell'import e funzionano comunque se torch manca.
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_reproducibility(seed: int = 42) -> None:
    """Imposta il seed di Python, NumPy e (se disponibile) PyTorch per esecuzioni riproducibili."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch è opzionale per i percorsi di codice puramente classici
        pass
