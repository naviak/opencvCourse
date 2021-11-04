import numpy as np
import cv2

block_size = 2
ksize = 3

maxfeatures = 100
qualitylevel = 1e-3


def showImgs(image):
    res = showImg(img)
    cv2.imshow('dst', res)

    pep = perspective(img)
    pep = ABImage(pep)
    res_pr = showImg(pep)
    cv2.imshow('rot', res_pr)


def set_block_size(x):
    global img
    global block_size
    block_size = max(x, 1)
    showImgs(img)


def set_maxfeatures(x):
    global img
    global maxfeatures
    maxfeatures = max(x, 1)
    showImgs(img)


def set_qualitylevel(x):
    global img
    global qualitylevel
    if not x:
        qualitylevel = 2
    else:
        qualitylevel = 1 / x

    showImgs(img)


def set_ksize(x):
    global img
    global ksize
    if not x % 2:
        ksize = max(x + 1, 1)
    else:
        ksize = max(x, 1)
    showImgs(img)


controlsH_window_name = 'controls Harris'
cv2.namedWindow(controlsH_window_name)

controlsGFT_window_name = 'controls GFT'
cv2.namedWindow(controlsGFT_window_name)

alpha = 80
beta = 5


def ABImage(image):
    im = np.copy(image)
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            for z in range(im.shape[2]):
                res = float(alpha) / 100 * im[x][y][z] + beta
                res = res if 0 <= res <= 255 else 255 if res > 255 else 0
                im[x][y][z] = res
    return im


def scaleImg(image, scale: float):
    width, height = image.shape[1], image.shape[0]
    return cv2.resize(image, (int(width * scale), int(height * scale)))


def showImg(image):
    res = np.copy(image)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, block_size, ksize, 0.04)

    corners = cv2.goodFeaturesToTrack(gray, maxfeatures, qualitylevel, 1)
    if corners is not None:
        for i in corners:
            x, y = i.ravel()
            cv2.circle(res, (int(x), int(y)), 3, 255, -1)
        res[dst > 0.01 * dst.max()] = [0, 0, 255]
    return res


def perspective(image):
    imag = np.copy(image)
    height, width = imag.shape[:2]
    center = (width / 2, height / 2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=45, scale=1)
    rotated_image = cv2.warpAffine(src=imag, M=rotate_matrix, dsize=(width, height))

    pts1 = np.float32([[0, 0], [150, 100], [0, 200], [200, 200]])
    pts2 = np.float32([[0, 0], [250, 150], [0, 100], [250, 250]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    new_image = cv2.warpPerspective(rotated_image, M, (200, 250))
    return new_image


filename = 'chessboard.jpg'
img = cv2.imread(filename)
img = scaleImg(img, 0.05)

cv2.createTrackbar('block size', controlsH_window_name, block_size, 20, set_block_size)
cv2.createTrackbar('ksize', controlsH_window_name, ksize, 20, set_ksize)

cv2.createTrackbar('max features', controlsGFT_window_name, maxfeatures, 1000, set_maxfeatures)
cv2.createTrackbar('qualitylevel', controlsGFT_window_name, 100, 100, set_qualitylevel)

print(img.shape)
showImg(img)

pep = perspective(img)
pep = ABImage(pep)
res_pr = showImg(pep)
cv2.imshow('rot', res_pr)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
