"""Genera esempi di inferenza + compliance e raccoglie automaticamente i casi di errore.

Salva gli overlay annotati in outputs/inference/, i candidati di errore in
outputs/failure_cases/, e stampa una distribuzione di compliance + un riepilogo
degli errori usato per compilare i report.
"""

import _bootstrap  # noqa: F401
import argparse
from collections import Counter

import cv2
import numpy as np

from src.compliance import assess_compliance, summarize_compliance
from src.config import CLASS_TO_ID, PERSON_CLASS, SH17_CLASSES, set_reproducibility
from src.data import label_path_for_image, list_images, read_yolo_label
from src.inference import load_model, predict_image
from src.postprocess import filter_detections
from src.visualization import draw_compliance, draw_detections

HELMET, VEST = CLASS_TO_ID["helmet"], CLASS_TO_ID["safety-vest"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--images", default="data/processed/sh17/images/val")
    ap.add_argument("--n-eval", type=int, default=200, help="immagini analizzate per statistiche/errori")
    ap.add_argument("--n-save", type=int, default=24, help="overlay di esempio da salvare")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    set_reproducibility(42)
    from pathlib import Path

    out_inf = Path("outputs/inference"); out_inf.mkdir(parents=True, exist_ok=True)
    out_fail = Path("outputs/failure_cases"); out_fail.mkdir(parents=True, exist_ok=True)

    model = load_model(args.weights)
    images = list_images(args.images)
    rng = np.random.default_rng(42)
    sample = [images[i] for i in rng.choice(len(images), size=min(args.n_eval, len(images)), replace=False)]

    compliance = Counter()
    saved_examples = saved_failures = 0
    fail_reasons = Counter()

    for img_path in sample:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        dets = filter_detections(predict_image(model, image, conf=args.conf, device=args.device),
                                 conf_threshold=args.conf)
        assessments = assess_compliance(dets)
        for a in assessments:
            compliance[a.status.value] += 1

        # Confronto con la ground truth per il rilevamento degli errori.
        gt = read_yolo_label(label_path_for_image(img_path))
        gt_person = sum(1 for b in gt if b.cls == PERSON_CLASS)
        gt_helmet = sum(1 for b in gt if b.cls == HELMET)
        pred_person = sum(1 for d in dets if d["cls"] == PERSON_CLASS)
        pred_helmet = sum(1 for d in dets if d["cls"] == HELMET)

        reason = None
        if gt_person and pred_person == 0:
            reason = "person_missed"
        elif gt_person and pred_person < gt_person:
            reason = "person_undercount"
        elif gt_helmet and pred_helmet == 0:
            reason = "helmet_missed"
        elif pred_person > gt_person and gt_person > 0:
            reason = "person_overcount"

        overlay = draw_compliance(draw_detections(image, dets, SH17_CLASSES), assessments)

        if reason and saved_failures < 20:
            fail_reasons[reason] += 1
            cv2.imwrite(str(out_fail / f"{reason}__{img_path.stem}.jpg"), overlay)
            saved_failures += 1
        elif saved_examples < args.n_save:
            cv2.imwrite(str(out_inf / f"{img_path.stem}_pred.jpg"), overlay)
            saved_examples += 1

    print("=== Distribuzione di compliance (lavoratori su", args.n_eval, "immagini) ===")
    print(dict(compliance), "| lavoratori totali:", sum(compliance.values()))
    print("=== Candidati di errore salvati:", saved_failures, "===")
    print("per motivo:", dict(fail_reasons))
    print("=== Overlay di esempio salvati:", saved_examples, "->", out_inf)


if __name__ == "__main__":
    main()
