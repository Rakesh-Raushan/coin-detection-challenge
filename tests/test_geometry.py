import pytest
import numpy as np
from app.services.geometry import CoinGeometry

def test_radius_calculation_square():
    width, height = 100, 100
    expected_radius = 50
    assert CoinGeometry.calculate_radius(width, height) == expected_radius

def test_radius_calculation_slanted():
    width, height = 120, 80
    expected_radius = 60 # since the max dimension is 120, the radius should be 120/2 = 60 (our assumption for slanted coins)
    assert CoinGeometry.calculate_radius(width, height) == expected_radius

def test_ellipse_params():
    bbox = [10, 20, 100, 50] # x, y, width, height
    params = CoinGeometry.get_ellipse_params(bbox)
    assert params['center'] == (60, 45) # center should be (x + w/2, y + h/2) = (10 + 50, 20 + 25)
    assert params['axes'] == (50, 25) # axes should be (w/2, h/2) = (100/2, 50/2)

def test_mask_generation_dimensions():
    bbox = [10, 10, 20, 20]
    image_shape = (100, 100)
    mask = CoinGeometry.generate_mask(bbox, image_shape)
    assert mask.shape == image_shape # mask should have the same height and width as the input image shape
    assert np.max(mask) == 255 # max value in the mask should be 255 (since it's a binary mask)
    assert np.min(mask) == 0 # min value in the mask should be 0