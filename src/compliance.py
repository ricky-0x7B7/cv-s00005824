"""Motore di compliance: associa i DPI rilevati ai lavoratori e assegna uno stato di sicurezza.

Questa è la logica applicativa personalizzata sovrapposta ai rilevamenti YOLO che
trasforma il rilevamento grezzo di oggetti in un sistema di monitoraggio della
sicurezza utilizzabile.

Regola (basata su regole, volutamente semplice e verificabile)::

    persona con casco E gilet di sicurezza    -> SAFE
    persona con esattamente uno tra casco/gilet -> PARTIALLY_SAFE
    persona senza nessuno dei due             -> UNSAFE

Un box di DPI viene associato a una persona quando il suo centro cade dentro il box
della persona OPPURE quando la sua IoU con il box della persona supera
``iou_threshold``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.config import ESSENTIAL_PPE, OPTIONAL_PPE, PERSON_CLASS
from src.postprocess import center_inside, iou_xyxy


class SafetyStatus(str, Enum):
    SAFE = "SAFE"
    PARTIALLY_SAFE = "PARTIALLY_SAFE"
    UNSAFE = "UNSAFE"


# Colori BGR (OpenCV) usati nel disegno dello stato di compliance.
STATUS_COLORS: dict[SafetyStatus, tuple[int, int, int]] = {
    SafetyStatus.SAFE: (0, 170, 0),         # verde
    SafetyStatus.PARTIALLY_SAFE: (0, 200, 255),  # ambra
    SafetyStatus.UNSAFE: (0, 0, 230),       # rosso
}


@dataclass
class WorkerAssessment:
    """Risultato di compliance per una singola persona rilevata."""

    person_box: tuple[float, float, float, float]
    status: SafetyStatus
    ppe_present: dict[str, bool] = field(default_factory=dict)

    @property
    def color(self) -> tuple[int, int, int]:
        return STATUS_COLORS[self.status]


def _ppe_belongs_to_person(
    ppe_box: tuple[float, float, float, float],
    person_box: tuple[float, float, float, float],
    iou_threshold: float,
) -> bool:
    return center_inside(ppe_box, person_box) or iou_xyxy(ppe_box, person_box) >= iou_threshold


def assess_compliance(
    detections: list[dict],
    iou_threshold: float = 0.05,
    essential_ppe: dict[str, int] | None = None,
    optional_ppe: dict[str, int] | None = None,
) -> list[WorkerAssessment]:
    """Assegna uno :class:`SafetyStatus` a ogni persona rilevata.

    Parameters
    ----------
    detections:
        Lista di dict, ciascuno con ``cls`` (id intero della classe) e ``xyxy`` (box in pixel).
    iou_threshold:
        IoU minima DPI/persona (in aggiunta al test del centro-interno) affinché il
        DPI conti come appartenente alla persona.
    """
    essential = essential_ppe if essential_ppe is not None else ESSENTIAL_PPE
    optional = optional_ppe if optional_ppe is not None else OPTIONAL_PPE
    id_to_name = {v: k for k, v in {**essential, **optional}.items()}

    persons = [d for d in detections if d["cls"] == PERSON_CLASS]
    ppe = [d for d in detections if d["cls"] in id_to_name]

    results: list[WorkerAssessment] = []
    for p in persons:
        pbox = tuple(p["xyxy"])
        present = {name: False for name in id_to_name.values()}
        for item in ppe:
            name = id_to_name[item["cls"]]
            if _ppe_belongs_to_person(tuple(item["xyxy"]), pbox, iou_threshold):
                present[name] = True

        n_essential = sum(present.get(name, False) for name in essential)
        if n_essential == len(essential):
            status = SafetyStatus.SAFE
        elif n_essential == 0:
            status = SafetyStatus.UNSAFE
        else:
            status = SafetyStatus.PARTIALLY_SAFE

        results.append(WorkerAssessment(person_box=pbox, status=status, ppe_present=present))
    return results


def summarize_compliance(assessments: list[WorkerAssessment]) -> dict[str, int]:
    """Conta i lavoratori per stato di sicurezza (comodo per dashboard / report)."""
    counts = {s.value: 0 for s in SafetyStatus}
    for a in assessments:
        counts[a.status.value] += 1
    counts["total_workers"] = len(assessments)
    return counts
