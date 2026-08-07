"""Unit tests for OpenVocabDetection utilities:  pytest test/test_detection.py"""
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lexicon.open_vocab_detector import OpenVocabDetection


def _det(box, phrase="obj", score=0.5):
    return OpenVocabDetection(phrase=phrase, score=score, box_xyxy=box)


def test_area_simple():
    d = _det([0, 0, 10, 20])
    assert math.isclose(d.area, 200.0)


def test_area_degenerate_box_is_zero():
    d = _det([5, 5, 5, 10])
    assert d.area == 0.0


def test_iou_identical_boxes_is_one():
    a = _det([0, 0, 10, 10])
    assert math.isclose(a.iou(a), 1.0)


def test_iou_disjoint_boxes_is_zero():
    a = _det([0, 0, 5, 5])
    b = _det([10, 10, 20, 20])
    assert a.iou(b) == 0.0


def test_iou_partial_overlap():
    a = _det([0, 0, 10, 10])   # area 100
    b = _det([5, 0, 15, 10])   # area 100, overlap 50
    # IoU = 50 / (100 + 100 - 50) = 50/150
    assert math.isclose(a.iou(b), 50.0 / 150.0)


def test_iou_is_symmetric():
    a = _det([0, 0, 10, 10])
    b = _det([3, 3, 12, 12])
    assert math.isclose(a.iou(b), b.iou(a))
