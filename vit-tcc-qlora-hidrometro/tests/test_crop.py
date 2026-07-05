"""Testes unitários do módulo de expansão de crop."""

from hidrometro.preprocessing.crop import (
    ExpansionRatios,
    coverage_ratio,
    crop_meter,
    expand_to_full_meter,
)
import numpy as np


def test_expand_to_full_meter_grows_bbox():
    bbox = (100, 100, 200, 200)
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    expanded = expand_to_full_meter(bbox, image.shape, ExpansionRatios(1.0, 1.0, 1.0, 1.0))
    assert expanded[0] < bbox[0]
    assert expanded[1] < bbox[1]
    assert expanded[2] > bbox[2]
    assert expanded[3] > bbox[3]


def test_coverage_ratio_is_one_for_valid_expansion():
    bbox = (100, 100, 200, 200)
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    expanded = expand_to_full_meter(bbox, image.shape)
    assert coverage_ratio(bbox, expanded) == 1.0


def test_crop_meter_returns_image():
    image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    bbox = (200, 150, 400, 350)
    crop, full, ratio = crop_meter(image, bbox)
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert ratio == 1.0
