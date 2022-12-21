import glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
import os

# connect seperated lines by gradient
    # https://stackoverflow.com/questions/63727525/how-to-connect-broken-lines-that-cannot-be-connected-by-erosion-and-dilation
    # https://stackoverflow.com/questions/43859750/how-to-connect-broken-lines-in-a-binary-image-using-python-opencv


# follow all connected pixels
# TODO
def group(img):
    # check status of each points
    # 0 : don't care, 1: line end, 2: part of line, <=3: intersection
    count = np.zeros(img.shape)
    pt_end = []
    pt_inter = []

    for j in range(1, img.shape[0] - 1):
        for i in range(1, img.shape[1] - 1):
            if img[j, i] == 0:
                continue

            count[j, i] = np.count_nonzero(img[j-1:j+2, i-1:i+2]) - 1
            if count[j, i] == 1:
                pt_end.append([j, i])
            if count[j, i] >= 3:
                pt_inter.append([j, i])

    # find lines from end to end or inter (check count==2 points)
    # each element of list lines are a list of grouped pixels
    lines_inter = [] # lines end at intersection point
    lines = [] # all lines

    for ends in pt_end:
        not_visited = np.ones(img.shape)
        temp_line = [[ends[0], ends[1]]]

        while(True):
            cur = temp_line[-1]
            not_visited[cur[0], cur[1]] = 0
            around = np.multiply(count[cur[0]-1:cur[0]+2, cur[1]-1:cur[1]+2], not_visited[cur[0]-1:cur[0]+2, cur[1]-1:cur[1]+2])
            next_pos = np.transpose(np.nonzero(around))
            
            # error handling: true only the assumption was false
            if len(next_pos) != 1 :
                print("multiple lines available")
                break
            
            # update line
            next_y = cur[0] + next_pos[0][0] - 1
            next_x = cur[1] + next_pos[0][1] - 1
            temp_line.append([next_y, next_x])
            not_visited[next_y, next_x] = 0
            
            # check end condition
            if count[next_y, next_x] >= 3:
                lines_inter.append(temp_line)
                break
            elif count[next_y, next_x] == 1:
                pt_end.remove([next_y, next_x])
                lines.append(temp_line)
                break
     
    # connect lines ended at intersection point
    for m in range(len(lines_inter)):
        for n in range(m+1, len(lines_inter)):
            line_m_dir = lines_inter[m][-1][1] - lines_inter[m][0][1]
            line_n_dir = lines_inter[n][-1][1] - lines_inter[n][0][1]
            if line_m_dir * line_n_dir < 0:
                lines.append(lines_inter[m] + lines_inter[n])
                    
    return lines


# classify line: use # of pixels, relevant position, etc
# remain only 3 longest lines and check positions
# TODO
def classify(lines):
    # order: life heart head?
    line_ordered = lines

    return line_ordered

if __name__ == "__main__":
    for img_path in glob.glob("line_sample/*.png"):
        img = cv.imread(img_path)
        # blur_img = cv.GaussianBlur(img, (3,3), 0) # less noise, but get connected lines
        skel_img = cv.cvtColor(skeletonize(img), cv.COLOR_BGR2GRAY)

        print(img_path)
        # step 1: connect seperated lines by gradient -> skip now
        lines = group(skel_img) # step 2
        lines = classify(lines) # step 3
        
        cv.imwrite("line_sample_result/skel_" + img_path.split(os.sep)[1], skel_img)