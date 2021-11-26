from cv2 import cv2 as cv
import numpy as np
from imutils import contours
from matplotlib import pyplot as plt
from functools import cmp_to_key
from bs4 import BeautifulSoup
import requests
time_region_width = 40


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


def find_reg(method, img, template):
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

    top_left, bottom_right, _ = find_reg(method, img, template)

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


def match_time_number(number, game=1):
    page = int(number / 20) + 1
    idx = number % 20
    page_url = "https://game-tournaments.com/dota-2/matches"
    headers = {'content-type': 'application/x-www-form-urlencoded; charset=UTF-8', 'x-requested-with': 'XMLHttpRequest'}

    raw_data = 'game=dota-2&rid=matches&ajax=block_matches_past&data%5Bs%5D=' + str(
        page) + '&data%5Btype%5D=gg&data%5Bscore%5D=0'

    page = requests.post(page_url, headers=headers, data=raw_data)

    soup = BeautifulSoup(page.content, "html.parser")
    job_elements = soup.find_all("span", class_='mbutton tresult')
    id = job_elements[idx]["data-mid"]
    print(f"Match id is {id}")

    return match_time_id(id, game)


def match_time_id(id, game_num=1):
    page_url = "https://game-tournaments.com/dota-2/bts-pro-series-season-9/sea/boom-vs-nigma-galaxy-sea-" + str(id)
    page = requests.get(page_url)
    soup = BeautifulSoup(page.content, "html.parser")

    job_elements = soup.find_all("a", class_="g-rezults")

    address = job_elements[0]["href"]
    image_url = 'https://en.game-tournaments.com' + address

    page = requests.get(image_url)
    image = np.asarray(bytearray(page.content), dtype="uint8")
    image = cv.imdecode(image, cv.IMREAD_COLOR)

    return math_time(image), image


id = 2

(time, region, boxes), img = match_time_number(id)

canvas = cv.cvtColor(region, cv.COLOR_GRAY2BGR)

for box in boxes:
    cv.rectangle(canvas, box.box[0], box.box[1], (200, 20, 20))

canvas = clipImg(canvas, 500)

print(f'Time of game {time}')

while True:

    cv.imshow('time', canvas)
    cv.imshow('img', img)

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break