import cv2
import numpy as np

cv2.namedWindow('img')

def thresh_callback(val):
    threshold = val
    cv2.imshow('g',gray)
    # Detect edges using Canny
    canny_output = cv2.Canny(gray, threshold, 100)
    # Find contours
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        cv2.drawContours(drawing, contours, i, (0, 255, 0), 1, cv2.LINE_AA, hierarchy, 1)
    # Show in a window
    cv2.imshow('img', cv2.add(img, drawing))



img = cv2.imread('image.jpg')
max_thresh = 255
thresh = 100
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.createTrackbar('threshold', 'img', 150, max_thresh, thresh_callback)

while True:
    k = cv2.waitKey(0)
    if k % 256 == 32:
        cv2.imwrite('img.jpg', img)
    if k % 256 == 27:
        pass
cv2.destroyAllWindows()
