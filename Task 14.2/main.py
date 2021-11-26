import requests
from bs4 import BeautifulSoup
from PIL import Image
import requests
import numpy as np
from io import BytesIO
from cv2 import cv2 as cv
import numpy as np
from imutils import contours
from matplotlib import pyplot as plt
from functools import cmp_to_key

time_region_width = 40

id_match = 413134
page_url = "https://game-tournaments.com/dota-2/d2cl-2021-season-5/group-stage/khan-vs-hydra-kir-"

page_url = page_url + str(id_match)

html = requests.get(page_url)

soup = BeautifulSoup(html.content, 'html.parser')

res = soup.find_all('a', 'g-rezults')[0]["href"]
res_url = 'https://game-tournaments.com' + res
response = requests.get(res_url)
img = np.asarray(bytearray(response.content), dtype='uint8')
print(img)


def clipImg(image, max_size):
    width, height = image.shape[1], image.shape[0]
    max_dim = max(width, height)
    ratio = float(max_size) / max_dim
    return cv.resize(image, (int(width * ratio), int(height * ratio)), interpolation=cv.INTER_CUBIC)


class Number:
    def __init__(self, num, box, isColon=False):
        self.num = num
        self.box = box
        self.isColon = isColon


def find_region(method, img, template):
    w, h = template.shape[::-1]

    res = cv.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    return top_left, bottom_right, res


def find_boxes(method, img, template, threshold=0.9):
    w, h = template.shape[::-1]

    res = cv.matchTemplate(img, template, method)

    detected_boxes = []

    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        detected_boxes.append((pt, (pt[0] + w, pt[1] + h)))

    return detected_boxes


def box_center(box):
    return (box[0][0] + box[1][0]) * 0.5


def compare(item1, item2):
    if box_center(item1.box) < box_center(item2.box):
        return -1
    elif box_center(item1.box) > box_center(item2.box):
        return 1
    else:
        return 0


def math_time(image):
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # where to find match time
    template = cv.imread('template.jpg', 0)

    # seems to work 'OK'
    method = eval('cv.TM_CCOEFF_NORMED')

    top_left, bottom_right, _ = find_region(method, img, template)

    time_region = img[top_left[1]:bottom_right[1], bottom_right[0]:bottom_right[0] + time_region_width]

    # numeric templates
    numbers = []
    for i in range(10):
        numbers.append(cv.imread('nums/' + str(i) + '.jpg', 0))
    colon_template = cv.imread('c.jpg', 0)

    registered_numbers = []

    for idx, number in enumerate(numbers):
        boxes = find_boxes(method, time_region, number)
        for box in boxes:
            registered_numbers.append(Number(idx, box))

    boxes = find_boxes(method, time_region, colon_template)
    for box in boxes:
        registered_numbers.append(Number(-1, box, True))

    registered_numbers = sorted(registered_numbers, key=cmp_to_key(compare))

    time = ''
    for num in registered_numbers:
        time += ':' if num.isColon else str(num.num)

    return time, time_region, registered_numbers


img = cv.imdecode(img, cv.IMREAD_COLOR)


time, region, boxes = math_time(img)

canvas = cv.cvtColor(region, cv.COLOR_GRAY2BGR)

for box in boxes:
    cv.rectangle(canvas, box.box[0], box.box[1], (200, 20, 20))

canvas = clipImg(canvas, 500)

print(time)

while True:

    cv.imshow('time', canvas)
    cv.imshow('img', img)

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
