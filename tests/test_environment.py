"""Smoke test per l'ambiente e la selezione del device."""


def test_core_imports():
    import cv2  # noqa: F401
    import numpy  # noqa: F401
    import sklearn  # noqa: F401
    import skimage  # noqa: F401


def test_select_device_returns_valid():
    from src.config import select_device

    assert select_device("auto") in {"mps", "cuda", "cpu"}
    assert select_device("cpu") == "cpu"


def test_sh17_classes_are_17_and_indexed():
    from src.config import CLASS_TO_ID, SH17_CLASSES

    assert len(SH17_CLASSES) == 17
    assert SH17_CLASSES[0] == "person"
    assert CLASS_TO_ID["helmet"] == 10
    assert CLASS_TO_ID["safety-vest"] == 16
