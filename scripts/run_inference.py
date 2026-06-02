"""Esegue rilevamento + compliance su un'immagine, una cartella o un video.

Uso::

    # singola immagine
    python scripts/run_inference.py --weights models/best.pt --source samples/example.jpg
    # cartella di immagini
    python scripts/run_inference.py --weights models/best.pt --source samples/
    # video
    python scripts/run_inference.py --weights models/best.pt --source clip.mp4
"""

import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import cv2

from src.compliance import assess_compliance, summarize_compliance
from src.config import SH17_CLASSES
from src.data import IMAGE_EXTS
from src.inference import load_model, predict_image, results_to_detections
from src.postprocess import filter_detections
from src.visualization import draw_compliance, draw_detections

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def annotate(image, detections, conf):
    detections = filter_detections(detections, conf_threshold=conf)
    out = draw_detections(image, detections, SH17_CLASSES)
    assessments = assess_compliance(detections)
    out = draw_compliance(out, assessments)
    return out, summarize_compliance(assessments)


def run_on_image(model, path: Path, out_dir: Path, conf: float, iou: float, imgsz: int, device: str):
    image = cv2.imread(str(path))
    if image is None:
        print(f"  saltata (illeggibile): {path}")
        return
    detections = predict_image(model, image, conf=conf, iou=iou, imgsz=imgsz, device=device)
    out, summary = annotate(image, detections, conf)
    out_path = out_dir / f"{path.stem}_pred.jpg"
    cv2.imwrite(str(out_path), out)
    print(f"  {path.name}: {summary} -> {out_path}")


def run_on_video(model, path: Path, out_dir: Path, conf: float, iou: float, imgsz: int, device: str):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = out_dir / f"{path.stem}_pred.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(source=frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        detections = results_to_detections(results[0])
        out, _ = annotate(frame, detections, conf)
        writer.write(out)
        n += 1
    cap.release()
    writer.release()
    print(f"  elaborati {n} fotogrammi -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferenza + compliance")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--source", required=True, help="Immagine, cartella o video")
    parser.add_argument("--output", default="outputs/inference")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.weights)
    source = Path(args.source)

    if source.is_dir():
        images = [p for p in sorted(source.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        print(f"Elaborazione di {len(images)} immagini da {source}")
        for p in images:
            run_on_image(model, p, out_dir, args.conf, args.iou, args.imgsz, args.device)
    elif source.suffix.lower() in VIDEO_EXTS:
        run_on_video(model, source, out_dir, args.conf, args.iou, args.imgsz, args.device)
    else:
        run_on_image(model, source, out_dir, args.conf, args.iou, args.imgsz, args.device)


if __name__ == "__main__":
    main()
