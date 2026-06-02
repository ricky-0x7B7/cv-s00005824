"""Test unitari per il motore di compliance e la geometria di post-processing."""

from src.compliance import SafetyStatus, assess_compliance, summarize_compliance
from src.config import CLASS_TO_ID
from src.postprocess import center_inside, filter_detections, iou_xyxy, nms

PERSON = CLASS_TO_ID["person"]
HELMET = CLASS_TO_ID["helmet"]
VEST = CLASS_TO_ID["safety-vest"]


def _det(cls, box, conf=0.9):
    return {"cls": cls, "conf": conf, "xyxy": box}


def test_iou_basic():
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) - (50 / 150)) < 1e-6


def test_center_inside():
    assert center_inside((4, 4, 6, 6), (0, 0, 10, 10))
    assert not center_inside((40, 40, 60, 60), (0, 0, 10, 10))


def test_nms_suppresses_overlap():
    import numpy as np

    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]])
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert 0 in keep and 2 in keep and 1 not in keep


def test_filter_detections_conf_and_area():
    dets = [_det(PERSON, (0, 0, 100, 100), conf=0.1), _det(PERSON, (0, 0, 100, 100), conf=0.9)]
    assert len(filter_detections(dets, conf_threshold=0.25)) == 1


def test_worker_safe_with_helmet_and_vest():
    person = (0, 0, 100, 200)
    dets = [_det(PERSON, person), _det(HELMET, (40, 5, 60, 30)), _det(VEST, (30, 80, 70, 140))]
    res = assess_compliance(dets)
    assert len(res) == 1
    assert res[0].status is SafetyStatus.SAFE


def test_worker_partially_safe_with_only_helmet():
    person = (0, 0, 100, 200)
    dets = [_det(PERSON, person), _det(HELMET, (40, 5, 60, 30))]
    assert assess_compliance(dets)[0].status is SafetyStatus.PARTIALLY_SAFE


def test_worker_unsafe_without_ppe():
    dets = [_det(PERSON, (0, 0, 100, 200))]
    assert assess_compliance(dets)[0].status is SafetyStatus.UNSAFE


def test_ppe_outside_person_not_associated():
    # Un casco lontano dalla persona non deve renderla sicura.
    person = (0, 0, 100, 200)
    far_helmet = (500, 500, 520, 520)
    dets = [_det(PERSON, person), _det(HELMET, far_helmet)]
    assert assess_compliance(dets)[0].status is SafetyStatus.UNSAFE


def test_summary_counts():
    dets = [
        _det(PERSON, (0, 0, 100, 200)), _det(HELMET, (40, 5, 60, 30)), _det(VEST, (30, 80, 70, 140)),
        _det(PERSON, (300, 0, 400, 200)),
    ]
    summary = summarize_compliance(assess_compliance(dets))
    assert summary["total_workers"] == 2
    assert summary["SAFE"] == 1
    assert summary["UNSAFE"] == 1
