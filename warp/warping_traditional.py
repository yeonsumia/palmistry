# reference
# https://medium.com/analytics-vidhya/hand-detection-and-finger-counting-using-opencv-python-5b594704eb08

import glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math
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
    # cv.imwrite("results/hull_" + img_path.split(os.sep)[1], img)
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
    pts = pts_order(pts)
    return pts


def pts_order(pts):
    # order = 0, 1, 2, 3 from last to first finger
    dist_largest = 0
    pair_03 = (0, 0)
    dist_table = np.zeros((4, 4))
    for i in range(4):
        for j in range(i + 1, 4):
            dist_table[i, j] = dist_table[j, i] = math.dist(pts[i], pts[j])
            if dist_table[i, j] > dist_largest:
                dist_largest = math.dist(pts[i], pts[j])
                pair_03 = (i, j)

    dist_small_0 = dist_largest
    pt_1 = pair_03[0]
    for j in range(4):
        if j in pair_03:
            continue
        if dist_table[pair_03[0], j] < dist_small_0:
            dist_small_0 = dist_table[pair_03[0], j]
            pt_1 = j

    pt_2 = pair_03[1]
    for j in range(4):
        if j in pair_03:
            continue
        if j == pt_1:
            continue
        pt_2 = j
        dist_small_3 = dist_table[pair_03[1], j]
    
    ordered_pts = []
    ordered_pts.append(pts[pair_03[0]])
    ordered_pts.append(pts[pt_1])
    ordered_pts.append(pts[pt_2])
    ordered_pts.append(pts[pair_03[1]])
    if dist_small_0 > dist_small_3:
        ordered_pts = ordered_pts[::-1]

    return ordered_pts


def get_5th_pt(pts):
    ratio = 1.5  # you may change this
    print(pts)
    slope = (pts[2][1] - pts[1][1]) / (pts[2][0] - pts[1][0])  # (y2-y1)/(x2-x1)
    finger_center = (int((pts[1][0] + pts[2][0]) / 2), int((pts[1][1] + pts[2][1]) / 2))
    if slope != 0:
        meet_y = (
            (pts[1][0] + pts[2][0] - 2 * pts[3][0]) * slope
            + (pts[1][1] + pts[2][1]) * slope**2
            + 2 * pts[3][1]
        ) / (2 * (1 + slope**2))
        meet_x = (
            (pts[1][1] + pts[2][1] - 2 * pts[3][1]) * slope
            + (pts[1][0] + pts[2][0])
            + 2 * pts[3][0] * slope**2
        ) / (2 * (1 + slope**2))
    else:
        meet_y = pts[3][1]
        meet_x = (pts[1][0] + pts[2][0]) / 2
    meet_y = int(meet_y)
    meet_x = int(meet_x)

    bot_y = int(meet_y + (meet_y - finger_center[1]) * ratio)
    bot_x = int(meet_x + (meet_x - finger_center[0]) * ratio)
    print(meet_x, meet_y)
    print(bot_x, bot_y)
    return [(bot_x, bot_y)]


def extract_points(img):
    thresh = skin_mask(img)
    img_contour, contours = contour(img, thresh)
    img_hull = convex_hull(img, contours)
    img_convexity, defects = convexity_defect(contours)
    pts = count_finger(img, contours, defects)
    pts += get_5th_pt(pts)
    cv.circle(img, pts[4], 10, [0, 0, 255], -1)
    # image outputs - why img_path works?
    cv.imwrite("results/thresh_" + img_path.split(os.sep)[1], thresh)
    cv.imwrite("results/contour_" + img_path.split(os.sep)[1], img_contour)
    cv.imwrite("results/hull_" + img_path.split(os.sep)[1], img_hull)
    cv.imwrite("results/convexity_" + img_path.split(os.sep)[1], img_convexity)
    return np.float32(pts)


def warp(img, pts):
    # test image size: c * r = 3024 * 4032
    r, c = img.shape[:2]
    # hardcoded
    pts = np.delete(pts, 1, axis=0)
    p = np.float32(
        [
            [951 / 3024 * c, 1987 / 4032 * r],
            [1598 / 3024 * c, 1783 / 4032 * r],
            [2161 / 3024 * c, 2330 / 4032 * r],
            [1555 / 3024 * c, 3265 / 4032 * r],
        ]
    )
    mtx = cv.getPerspectiveTransform(pts, p)
    dst = cv.warpPerspective(img, mtx, (c, r))
    return dst


if __name__ == "__main__":
    for img_path in glob.glob("data/*.jpg"):
        img = cv.imread(img_path)
        pts = extract_points(img)
        cv.imwrite("results/out_" + img_path.split(os.sep)[1], img)
        # if points are not extracted well, skip that case
        if pts.shape[0] != 5:
            continue
        warped = warp(img, pts)
        cv.imwrite("results/warped_" + img_path.split(os.sep)[1], warped)
