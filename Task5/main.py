import cv2
import numpy as np

lowVal = 0
highVal = 0
noiseArr = []
imgNumber = 10

def setImg(x):
    global imgNumber
    imgNumber = x
    cv2.imshow('noise', getNoiseImg(img))
    for i in range(imgNumber):
        noiseArr.append(getNoiseImg(img))
    mat = np.array(noiseArr)
    n = np.average(mat, axis=0)
    cv2.imshow('made', n.astype(np.uint8))


def getNoiseImg(img):
    noise = np.zeros(img.shape, np.int8)
    cv2.randu(noise, lowVal, highVal)
    buf = cv2.add(img, noise, dtype=cv2.CV_8UC3)
    return buf

def setlowval(x):
    noiseArr = []
    global lowVal
    lowVal = x
    cv2.imshow('noise', getNoiseImg(img))
    for i in range(imgNumber):
        noiseArr.append(getNoiseImg(img))
    mat = np.array(noiseArr)
    n = np.average(mat, axis=0)
    cv2.imshow('made', n.astype(np.uint8))


def sethighval(x):
    noiseArr = []
    global highVal
    highVal = x
    cv2.imshow('noise', getNoiseImg(img))
    for i in range(imgNumber):
        noiseArr.append(getNoiseImg(img))
    mat = np.array(noiseArr)
    n = np.average(mat, axis=0)
    cv2.imshow('made', n.astype(np.uint8))

img = cv2.cvtColor(cv2.imread('lena.jpg'), cv2.COLOR_BGR2GRAY)
# mask = cv2.inRange(imgHsv, lower, upper)
cv2.imshow('im',img)

cv2.createTrackbar('lowval', 'im', 0, 255, setlowval)
cv2.createTrackbar('highval', 'im', 0, 255, sethighval)
cv2.createTrackbar('imgNum', 'im', 5, 100, setImg)

while True:
    k = cv2.waitKey(0)
    if k % 256 == 32:
        cv2.imwrite('img.jpg', img)
    if k % 256 == 27:
        pass
cv2.destroyAllWindows()
