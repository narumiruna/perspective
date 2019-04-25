# perspective

## code
transform point
```python
def perspective_point(point, startpoints, endpoints):
    coeffs = _get_perspective_coeffs(endpoints, startpoints)

    a, b, c, d, e, f, g, h = coeffs

    x, y = point

    return [(a * x + b * y + c) / (g * x + h * y + 1),
            (d * x + e * y + f) / (g * x + h * y + 1)]
```

## before
![](sample.jpg)

## after
![](perspective.jpg)

## references
- https://github.com/pytorch/vision/pull/781/files
- https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.transform
- https://docs.opencv.org/master/da/d54/group__imgproc__transform.html
