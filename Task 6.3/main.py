import cv2
import numpy as np
import imutils


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
        elif 4 < len(approx) < 8:
            shape = "oval"
        else:
            shape = "circle"
        return shape


cv2.namedWindow('img')

img = cv2.imread('shapes.jpg')
resized = imutils.resize(img, width=600)
ratio = img.shape[0] / float(resized.shape[0])
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

canny_output = cv2.Canny(gray, 60, 255)
# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(canny_output.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(cnts) == 2:
    cnts = cnts[0]

elif len(cnts) == 3:
    cnts = cnts[1]

sd = ShapeDetector()

for c in cnts:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape = sd.detect(c)
    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
    cv2.putText(img, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)
    # show the output image
cv2.imshow("Image", img)

while True:
    k = cv2.waitKey(0)
    if k % 256 == 32:
        cv2.imwrite('img.jpg', img)
    if k % 256 == 27:
        pass
cv2.destroyAllWindows()