import numpy as np
import pytest
from deepplate.enhancement.preprocessor import ImageEnhancer, EnhancerConfig


def make_image(h=60, w=200):
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def test_enhance_returns_same_channels():
    img = make_image()
    enhancer = ImageEnhancer()
    out = enhancer.enhance(img)
    assert out.ndim == 3
    assert out.shape[2] == 3


def test_upscale_doubles_size():
    img = make_image(60, 200)
    enhancer = ImageEnhancer(EnhancerConfig(upscale_factor=2, denoise=False))
    out = enhancer.enhance(img)
    assert out.shape[0] == 120
    assert out.shape[1] == 400


def test_no_upscale():
    img = make_image(60, 200)
    enhancer = ImageEnhancer(EnhancerConfig(upscale_factor=1, denoise=False))
    out = enhancer.enhance(img)
    assert out.shape[:2] == (60, 200)
