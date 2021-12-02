from cv2 import cv2 as cv
import numpy as np
from imutils import contours
from matplotlib import pyplot as plt
from functools import cmp_to_key
from bs4 import BeautifulSoup
import requests

time_region_width = 40

cap = cv.VideoCapture('video/traffic.mp4')

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = np.empty([0, 1, 2], dtype=np.float32)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# add points to the existing array on mouseclick
user_points = np.empty([0, 1, 2], dtype=np.float32)


def sample_track_points(event, x, y, flags, param):
    global user_points
    if event == cv.EVENT_LBUTTONDBLCLK:
        user_points = np.empty([1, 1, 2], dtype=np.float32)
        user_points[0][0] = [x, y]


# set the mouse call back
cv.namedWindow("frame")
cv.setMouseCallback("frame", sample_track_points)

# start the processing
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.release()
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if len(user_points) > 0:
        p0 = np.concatenate([p0, user_points])
        user_points = np.empty([0, 1, 2])
    print(p0.size)
    if p0.size != 0:
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv.add(frame, mask)
        cv.imshow('frame', img)
        p0 = good_new.reshape(-1, 1, 2)

    else:
        cv.imshow('frame', frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()

cv.destroyAllWindows()
cap.release()
