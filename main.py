from PIL import Image, ImageDraw

import functional as F
import transforms
from utils import load_json


def main():
    # load image and label
    data = load_json('sample.json')
    img = Image.open(data['imagePath'])

    # transform image
    w, h = img.size
    startpoints, endpoints = transforms.RandomPerspective.get_params(w, h, 0.5)
    new_img = F.perspective(img, startpoints, endpoints)

    # transform points
    shapes = data['shapes']
    for shape in shapes:
        shape['points'] = [tuple(F.perspective_point(point, startpoints, endpoints)) for point in shape['points']]

    # draw polygon
    draw = ImageDraw.Draw(new_img)
    for shape in shapes:
        draw.polygon(shape['points'], fill='blue')

    # show result
    new_img.show()

if __name__ == "__main__":
    main()
