import numpy as np
import pytest

from wastesorter.evaluate import choose_threshold, selective

CONFIDENCES = np.array([0.95, 0.90, 0.80, 0.60, 0.40])
CORRECT = np.array([True, True, True, False, False])


def test_selective_reports_coverage_and_accuracy_of_answered_images():
    point = selective(CONFIDENCES, CORRECT, threshold=0.5)

    assert point.coverage == pytest.approx(0.8)
    assert point.accuracy == pytest.approx(0.75)


def test_selective_with_nothing_answered_has_zero_coverage():
    point = selective(CONFIDENCES, CORRECT, threshold=0.99)

    assert (point.coverage, point.accuracy) == (0.0, 0.0)


def test_threshold_is_the_lowest_one_reaching_the_target():
    chosen = choose_threshold(CONFIDENCES, CORRECT, target_accuracy=1.0)

    assert chosen.threshold == pytest.approx(0.61)
    assert chosen.coverage == pytest.approx(0.6)


def test_threshold_is_zero_when_everything_already_meets_the_target():
    chosen = choose_threshold(CONFIDENCES, CORRECT, target_accuracy=0.5)

    assert chosen.threshold == 0.0
    assert chosen.coverage == 1.0


def test_unreachable_target_falls_back_to_the_most_accurate_threshold():
    correct = np.array([False, True, True, False, False])

    chosen = choose_threshold(CONFIDENCES, correct, target_accuracy=1.0)

    assert chosen.accuracy == pytest.approx(2 / 3)
