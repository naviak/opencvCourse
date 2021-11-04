import argparse
import imutils
import cv2
import numpy as np
from scipy.spatial import distance

# load the image, convert it to grayscale, blur it slightly,
# and threshold it
image = cv2.imread('shapes.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
imgshape = np.asarray(image.shape[:2])
for c in cnts:
    # compute the center of the contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # draw the contour and center of the shape on the image
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(image, "center", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    center = np.array([cX, cY])

    dists = np.array([distance.minkowski(x, center) for x in c])
    arg = np.argmax(dists)
    point = c[arg][0]
    cv2.line(image, center, point, (255, 255, 0), 3)

    if np.any(4 * dists[arg] > imgshape):
        cv2.putText(image, "BIG", (cX + 20, cY + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        cv2.putText(image, "small", (cX + 20, cY + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # show the image

cv2.imshow("Image", image)
cv2.waitKey(0)
