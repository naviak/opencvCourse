import scipy.spatial
from cv2 import cv2
import numpy as np
import scipy
from imutils import contours


def clipImg(image, max_size):
    width, height = image.shape[1], image.shape[0]
    max_dim = max(width, height)
    ratio = float(max_size) / max_dim
    return cv2.resize(image, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_CUBIC)


paths = [
    'kl.jpg',
    "plnt.jpg",
]

img = cv2.imread(paths[0])
img = clipImg(img, 600)

controlls_window_name = 'controlls'
edges_window_name = 'edges'
cv2.namedWindow(controlls_window_name)
cv2.namedWindow(edges_window_name)


def pass_callback(x):
    pass


treshold1 = 100
treshold2 = 100


def set_treshold1(x):
    global treshold1
    treshold1 = x


def set_treshold2(x):
    global treshold2
    treshold2 = x


cv2.createTrackbar('treshold 1', controlls_window_name, treshold1, 800, set_treshold1)
cv2.createTrackbar('treshold 2', controlls_window_name, treshold2, 800, set_treshold2)

canvas = np.zeros(img.shape, dtype=np.uint8)
overlay = np.zeros(img.shape, dtype=np.uint8)
approx_cntrs = np.zeros(img.shape, dtype=np.uint8)


def draw_bounding_boxes(canvas):
    for c in contours:
        draw_countour_bounds(c, canvas)


def draw_countour_bounds(c, canvas, color=(0, 255, 0)):
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)


x_pos = 1
y_pos = 1


def onMouse(event, x, y, flags, param):
    global x_pos, y_pos
    x = np.clip(x, 0, img.shape[1] - 1)
    y = np.clip(y, 0, img.shape[0] - 1)

    x_pos = x
    y_pos = y


cv2.setMouseCallback(edges_window_name, onMouse)

total_area = img.shape[0] * img.shape[1]

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 0, 0)
thickness = 2


def filter_contours(cntrs, tresh=100):
    filtered = []
    approxes = []
    for c in cntrs:
        hull = cv2.convexHull(c)

        area = cv2.contourArea(hull)
        approx = approx_contour(c)
        if area > tresh:
            filtered.append(c)
            approxes.append(approx)
    return filtered, approxes


def approx_contour(cntr):
    perimeter = cv2.arcLength(cntr, True)
    approx = cv2.approxPolyDP(cntr, 0.02 * perimeter, True)
    return approx


def vect_len(v):
    v = np.reshape(v, -1)
    return np.sqrt(v[0] ** 2 + v[1] ** 2)


def getPt(x):
    x = np.reshape(x, -1)
    return [x[0], x[1]]


def diff(x, y):
    x = x.ravel()
    y = y.ravel()
    return [x[0] - y[0], x[1] - y[1]]


def classify_shape(cntr, approx):
    if len(approx) == 3:
        shape = "triangle"

    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        fig_area = cv2.contourArea(approx)
        approx_area = vect_len(diff(approx[0], approx[1])) ** 2

        ar = fig_area / approx_area
        shape = "square" if 0.9 <= ar <= 1.2 else "rectangle"

    else:
        M = cv2.moments(approx)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        radius = vect_len(approx[0] - (cX, cY))

        approx_area = np.pi * (radius ** 2)
        fig_area = cv2.contourArea(approx)
        ar = approx_area / fig_area

        if 0.9 <= ar <= 1.2:
            shape = "circle"
        elif 0.5 <= ar <= 2.0:
            shape = "oval"
        else:
            shape = "unknown"
    # return the name of the shape
    return shape


while True:
    blurred = cv2.blur(img, (3, 3))
    edges_canny = cv2.Canny(blurred, treshold1, treshold2)

    contours, hierarchy = cv2.findContours(edges_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours, approxs = filter_contours(contours)
    canvas[:, :, :] = 0
    overlay[:, :, :] = 0
    approx_cntrs[:, :, :] = 0

    draw_bounding_boxes(canvas)

    cv2.drawContours(overlay, contours, -1, (50, 255, 0), 2)

    for idx, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        # point in bounding box
        ds = [scipy.spatial.distance.minkowski(c[i], c[j]) for i in range(len(c) - 1) for j in range(i, len(c))]
        ln = np.max(ds)
        hull = cv2.convexHull(c)

        area = cv2.contourArea(hull)

        fig_type = 'unknown'
        fig_type = classify_shape(c, approxs[idx])
        draw_countour_bounds(c, canvas, (244, 0, 0))
        area_ratio = area * 100 / total_area
        cv2.putText(canvas, 'area: ' + "{:.1f}".format(area_ratio) + '%',
                   (np.int32(x + w / 2 - 40), np.int32(y + h / 2)), font, 0.6, fontColor, thickness)

        size_label = 'small'
        if ln < max(canvas.shape[0], canvas.shape[1]) / 4:
            size_label = 'small'
        else:
            size_label = 'large'

        cv2.putText(canvas, size_label,
                   (np.int32(x + w / 2 - 40), np.int32(y + h / 2) + 15), font, 0.6, fontColor, thickness)

        cv2.putText(canvas, 'type: ' + fig_type,
                   (np.int32(x + w / 2 - 40), np.int32(y + h / 2 + 30)), font, 0.6, fontColor, thickness)

    cv2.imshow('img', cv2.add(img, overlay))

    result = canvas
    result = cv2.add(cv2.cvtColor(edges_canny, cv2.COLOR_GRAY2BGR), canvas)

    cv2.imshow(edges_window_name, result)

    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break
