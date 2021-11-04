import cv2 as cv
import numpy as np


def clipImg(image, max_size):
    width, height = image.shape[1], image.shape[0]
    max_dim = max(width, height)
    ratio = float(max_size) / max_dim
    return cv.resize(image, (int(width * ratio), int(height * ratio)), interpolation=cv.INTER_CUBIC)


def showImgs():
    global img
    overlay = np.zeros(img.shape, dtype=np.uint8)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, gray.shape[0] / dist,
                              param1=tresh_param1, param2=tresh_param2,
                              minRadius=50, maxRadius=65)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(overlay, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(overlay, center, radius, (255, 0, 255), 3)

    output = cv.add(img, overlay)
    cv.imshow('img', output)
    cv.imshow('lines', overlay)


img = cv.imread("money.png")
img = clipImg(img, 600)

controls_window_name = 'controlls'
cv.namedWindow(controls_window_name)

tresh_param1 = 24
tresh_param2 = 27
dist = 5


def set_param1(x):
    global tresh_param1
    tresh_param1 = x + 1
    showImgs()


def set_param2(x):
    global tresh_param2
    tresh_param2 = x + 1
    showImgs()


def set_dist(x):
    global dist
    dist = max(1, x)
    showImgs()


cv.createTrackbar('thd 1', controls_window_name, tresh_param1, 500, set_param1)
cv.createTrackbar('thd 2', controls_window_name, tresh_param2, 500, set_param2)
cv.createTrackbar('dist', controls_window_name, dist, 20, set_dist)
showImgs()

while True:
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
