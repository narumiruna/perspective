"""
https://github.com/surgan12/vision/blob/cad877403177961e5f289274ed4226f6466be446/torchvision/transforms/transforms.py
"""

from __future__ import division

import random

from PIL import Image

import functional as F


class RandomPerspective(object):
    """Performs Perspective transformation of the given PIL Image randomly with a given probability.
    Args:
        interpolation : Default- Image.BICUBIC
        p (float): probability of the image being perspectively transformed. Default value is 0.5
        distortion_scale(float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.
    """

    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=Image.BICUBIC):
        self.p = p
        self.interpolation = interpolation
        self.distortion_scale = distortion_scale

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be Perspectively transformed.
        Returns:
            PIL Image: Random perspectivley transformed image.
        """
        if not F._is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        if random.random() < self.p:
            width, height = img.size
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            return F.perspective(img, startpoints, endpoints, self.interpolation)
        return img

    @staticmethod
    def get_params(width, height, distortion_scale):
        """Get parameters for ``perspective`` for a random perspective transform.
        Args:
            width : width of the image.
            height : height of the image.
        Returns:
            List containing [top-left, top-right, bottom-right, bottom-left] of the orignal image,
            List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        half_height = int(height / 2)
        half_width = int(width / 2)
        topleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(0, int(distortion_scale * half_height)))
        topright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(0, int(distortion_scale * half_height)))
        botright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        botleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
