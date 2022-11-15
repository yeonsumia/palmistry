# reference
# https://medium.com/analytics-vidhya/hand-detection-and-finger-counting-using-opencv-python-5b594704eb08

import glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os


def skin_mask(img):
    # hsvim : blue, green, red -> hue, saturation, value
    hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower = np.array([0, 20, 80], dtype="uint8")
    upper = np.array([50, 255, 255], dtype="uint8")
    skinRegionHSV = cv.inRange(hsvim, lower, upper)

    # blur image to improve masking
    blurred = cv.blur(skinRegionHSV, (2, 2))

    ret, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY)
    # thresh = cv.adaptiveThreshold(blurred,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    return thresh


def contour(img, thresh):
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv.contourArea(x))
    cv.drawContours(img, [contours], -1, (255, 255, 0), 2)
    return img, contours


def convex_hull(img, contours):
    hull = cv.convexHull(contours)
    cv.drawContours(img, [hull], -1, (0, 255, 255), 2)
    #cv.imwrite("results/hull_" + img_path.split(os.sep)[1], img)
    return img


def convexity_defect(contours):
    hull = cv.convexHull(contours, returnPoints=False)
    defects = cv.convexityDefects(contours, hull)
    for i in range(defects.shape[0]):
      sp, ep, fp, dist = defects[i, 0]
      start = tuple(contours[sp][0])
      end = tuple(contours[ep][0])
      far = tuple(contours[fp][0])
      cv.circle(img, far, 10, [255, 0, 255], -1)
    return img, defects


def count_finger(img, contours, defects):
    if defects is not None:
        pts = []
    for i in range(defects.shape[0]):  # calculate the angle
        s, e, f, d = defects[i][0]
        start = tuple(contours[s][0])
        end = tuple(contours[e][0])
        far = tuple(contours[f][0])
        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = np.arccos(
            (b**2 + c**2 - a**2) / (2 * b * c)
        )  #      cosine theorem
        if angle <= (5 / 12) * np.pi:  # angle less than 90 degree, treat as fingers
            cv.circle(img, far, 10, [0, 0, 255], -1)
            pts.append(far)
    return pts


def extract_points(img):
    thresh = skin_mask(img)
    img_contour, contours = contour(img, thresh)
    img_hull = convex_hull(img, contours)
    img_convexity, defects = convexity_defect(contours)
    pts = count_finger(img, contours, defects)

    # image outputs - why img_path works?
    #cv.imwrite("results/thresh_" + img_path.split(os.sep)[1], thresh)
    #cv.imwrite("results/contour_" + img_path.split(os.sep)[1], img_contour)
    #cv.imwrite("results/hull_" + img_path.split(os.sep)[1], img_hull)
    #cv.imwrite("results/convexity_" + img_path.split(os.sep)[1], img_convexity)
    return np.float32(pts)


def warp(img, pts):
    r, c = img.shape[:2]
    p = np.float32([[1282, 1808], [951, 1987], [2161, 2330], [1598, 1783]])
    mtx = cv.getPerspectiveTransform(pts, p)
    dst = cv.warpPerspective(img, mtx, (c, r))
    return dst


if __name__ == "__main__":
    for img_path in glob.glob("data/*.jpg"):
        img = cv.imread(img_path)
        pts = extract_points(img)
        cv.imwrite("results/out_" + img_path.split(os.sep)[1], img)
        # if points are not extracted well, skip that case
        if pts.shape[0] != 4:
            continue
        warped = warp(img, pts)
        cv.imwrite("results/warped_" + img_path.split(os.sep)[1], warped)
