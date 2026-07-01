import numpy as np

from src.postprocess import add_bloom


def test_zero_strength_is_noop():
    img = np.random.rand(20, 30, 3)
    assert np.array_equal(add_bloom(img, 0.0), img)


def test_bloom_adds_energy_without_removing_any():
    img = np.zeros((41, 41, 3))
    img[20, 20] = [5.0, 5.0, 5.0]
    out = add_bloom(img, 1.0)
    assert out.sum() > img.sum()
    assert np.all(out >= img - 1.0e-9)


def test_bloom_spreads_to_neighbors():
    img = np.zeros((41, 41, 3))
    img[20, 20] = [10.0, 0.0, 0.0]
    out = add_bloom(img, 1.0)
    assert out[20, 21, 0] > 0.0
    assert out[19, 20, 0] > 0.0
