import numpy as np
import cv2

block_size = 2
ksize = 3

maxfeatures = 100
qualitylevel = 1e-3


def set_block_size(x):
    global block_size
    block_size = max(x, 1)
    showImg()


def set_maxfeatures(x):
    global maxfeatures
    maxfeatures = max(x, 1)
    showImg()


def set_qualitylevel(x):
    global qualitylevel
    if not x:
        qualitylevel = 2
    else:
        qualitylevel = 1 / x


def set_ksize(x):
    global ksize
    if not x % 2:
        ksize = max(x + 1, 1)
    else:
        ksize = max(x, 1)
    showImg()


controlsH_window_name = 'controls Harris'
cv2.namedWindow(controlsH_window_name)

controlsGFT_window_name = 'controls GFT'
cv2.namedWindow(controlsGFT_window_name)


def scaleImg(image, scale: float):
    width, height = image.shape[1], image.shape[0]
    return cv2.resize(image, (int(width * scale), int(height * scale)))


def showImg():
    global img
    res = np.copy(img)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, block_size, ksize, 0.04)

    corners = cv2.goodFeaturesToTrack(gray, maxfeatures, qualitylevel, 1)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(res, (int(x), int(y)), 3, 255, -1)
    # result is dilated for marking the corners, not importan
    # Threshold for an optimal value, it may vary depending on the image.
    res[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imshow('dst', res)


filename = 'chessboard.jpg'
img = cv2.imread(filename)
img = scaleImg(img, 0.2)

cv2.createTrackbar('block size', controlsH_window_name, block_size, 20, set_block_size)
cv2.createTrackbar('ksize', controlsH_window_name, ksize, 20, set_ksize)

cv2.createTrackbar('max features', controlsGFT_window_name, maxfeatures, 1000, set_maxfeatures)
cv2.createTrackbar('qualitylevel', controlsGFT_window_name, 100, 100, set_qualitylevel)
showImg()

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
