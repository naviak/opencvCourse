from cv2 import cv2 as cv
import numpy as np
from imutils import contours
from matplotlib import pyplot as plt
from functools import cmp_to_key
from bs4 import BeautifulSoup
import requests
time_region_width = 40


cap = cv.VideoCapture('video/bmw.mp4')

fgbg = cv.createBackgroundSubtractorKNN()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    cv.imshow('frame',fgmask)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()