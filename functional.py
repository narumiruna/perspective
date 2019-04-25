"""
https://github.com/surgan12/vision/blob/cad877403177961e5f289274ed4226f6466be446/torchvision/transforms/functional.py
"""
from __future__ import division

import torch
from PIL import Image, ImageEnhance, ImageOps


def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _get_perspective_coeffs(startpoints, endpoints):
    """Helper function to get the coefficients (a, b, c, d, e, f, g, h) for the perspective transforms.
    In Perspective Transform each pixel (x, y) in the orignal image gets transformed as,
     (x, y) -> ( (ax + by + c) / (gx + hy + 1), (dx + ey + f) / (gx + hy + 1) )
    Args:
        List containing [top-left, top-right, bottom-right, bottom-left] of the orignal image,
        List containing [top-left, top-right, bottom-right, bottom-left] of the transformed
                   image
    Returns:
        octuple (a, b, c, d, e, f, g, h) for transforming each pixel.
    """
    matrix = []

    for p1, p2 in zip(endpoints, startpoints):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = torch.tensor(matrix, dtype=torch.float)
    B = torch.tensor(startpoints, dtype=torch.float).view(8)
    res = torch.gels(B, A)[0]
    return res.squeeze_(1).tolist()


def perspective(img, startpoints, endpoints, interpolation=Image.BICUBIC):
    """Perform perspective transform of the given PIL Image.
    Args:
        img (PIL Image): Image to be transformed.
        coeffs (tuple) : 8-tuple (a, b, c, d, e, f, g, h) which contains the coefficients.
                            for a perspective transform.
        interpolation: Default- Image.BICUBIC
    Returns:
        PIL Image:  Perspectively transformed Image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    coeffs = _get_perspective_coeffs(startpoints, endpoints)
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, interpolation)

def perspective_point(point, startpoints, endpoints):
    coeffs = _get_perspective_coeffs(endpoints, startpoints)

    a, b, c, d, e, f, g, h = coeffs

    x, y = point

    return [(a * x + b * y + c) / (g * x + h * y + 1),
            (d * x + e * y + f) / (g * x + h * y + 1)]
