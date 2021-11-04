import cv2
import numpy as np
import random


def putTextImage(im, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 400)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(im, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)


image = cv2.imread("lena.jpg")
image = cv2.resize(image,(400,400))

gaus_blur = cv2.blur(image, (3, 3))
median_blur = cv2.medianBlur(image, 3)

kernel = np.array([[0.0, -1.0, 0.0],
                   [-1.0, 5.0, -1.0],
                   [0.0, -1.0, 0.0]])
kernel = kernel / np.sum(kernel)
ker_filter = cv2.filter2D(image, -1, kernel)
sobelx = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=3)
sobely = cv2.Sobel(image, cv2.CV_8U, 0, 1, ksize=3)
sobelxy = cv2.Sobel(image, cv2.CV_8U, 1, 1, ksize=3)

laplas = cv2.Laplacian(image, cv2.CV_8U, ksize=3)

transforms = {"orig": image, "gaus_blur": gaus_blur, "medianBlur": median_blur,
              "filter": ker_filter, "sobelx": sobelx, "sobely": sobely,
              "sobelxy": sobelxy, "laplas": laplas}

for key, value in transforms.items():
    putTextImage(value, key)
d1 = {key: value for i, (key, value) in enumerate(transforms.items()) if i < len(transforms) / 2}

d2 = {key: value for i, (key, value) in enumerate(transforms.items()) if i >= len(transforms) / 2}
print(len(transforms))

h = np.hstack(tuple(d1.values()))
l = np.hstack(tuple(d2.values()))
res = np.vstack((h, l))
cv2.imshow('key', res)

cv2.waitKey(0)
cv2.destroyAllWindows()
