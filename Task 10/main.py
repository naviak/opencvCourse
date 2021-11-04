import numpy as np
import cv2


def scaleImg(image, scale: float):
    width, height = image.shape[1], image.shape[0]
    return cv2.resize(image, (int(width * scale), int(height * scale)))


img1 = cv2.imread('1.png')
img1 = scaleImg(img1, 1.05)
img2 = cv2.imread('2.png')

orb = cv2.SIFT()
orb = orb.create(400)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)

cv2.imshow("res", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
