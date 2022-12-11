import os
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
import mediapipe as mp


#########################################################################################
# Sketch of idea                                                                        #
# - Build a graph : nodes = end pt. + intersection pt.                                  #
#                  edges = line segments (save pixel coordinates with next direction)   #
# - Find all possible paths(lines) from graph : graph backtracking                      #
# - Find three lines nearest to each cluster centers in feature space                   #
#   Find cluster centers using k-means clustering (k=3)                                 #
#########################################################################################
# TODO                                                                                  #
# - Consider not connected lines                                                        #
# - Find best descriptors(features)                                                     #
# - Reduce running time                                                                 #
# - Corner cases(close intersection pts)                                                #
#########################################################################################


### 0. Load Data ###

# PLSU folder should exist in advance
# rectify images in PLSU folder
# find homography matrix using original image
# then apply homography matrix to detected line image
# input is the number 'idx' from image{idx}.jpg(.png)
def rectify(idx):
    img_path = './PLSU/PLSU/'
    image = cv.imread(img_path + 'img/image' + str(idx) +'.jpg')
    image_mask = cv.imread(img_path + 'Mask/image' + str(idx) + '.png', cv.IMREAD_GRAYSCALE)
    mp_hands = mp.solutions.hands

    # 7 landmark points (normalized)
    pts_index = [0,1,2,5,9,13,17]
    pts_target_normalized = np.float32([[0.47036204, 0.864946  ],
                                        [0.29357246, 0.7761536 ],
                                        [0.19689673, 0.64541924],
                                        [0.40376574, 0.4919375 ],
                                        [0.52068794, 0.48719874],
                                        [0.61768305, 0.5140611 ],
                                        [0.7081194 , 0.5679793 ]])
    
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks == None: return np.zeros_like(image)
        image_height, image_width, _ = image.shape
        hand_landmarks = results.multi_hand_landmarks[0]
        pts = np.float32([[hand_landmarks.landmark[i].x*image_width,
                           hand_landmarks.landmark[i].y*image_height] for i in pts_index])
        pts_target = np.float32([[x*image_width, y*image_height] for x,y in pts_target_normalized])
        M, mask = cv.findHomography(pts, pts_target, cv.RANSAC,5.0)
        rectified_image = cv.warpPerspective(image_mask, M, (image_width, image_height))
        return rectified_image

# load rectified data from PLSU to new folder
def load_data(num_data):
    # make directory
    data_path = './line_sample'
    result_path = './line_sample_result'
    path_list = [data_path, result_path]
    for path in path_list:
        os.makedirs(path, exist_ok=True)
        
    # count the current number of data
    cur_num_data = len(os.listdir(data_path))
    
    # load rectified data from PLSU data
    offset = 50
    img_path_list = [img_path for img_path in glob.glob('./PLSU/PLSU/Img/*')][offset+cur_num_data:offset+num_data]
    for i,img_path in enumerate(img_path_list):
        idx = img_path.split('image')[1].split('.')[0]
        rectified_image = rectify(idx)
        if np.sum(rectified_image)==0: continue
        cv.imwrite('line_sample/image' + str(i+cur_num_data) + '.png', rectified_image)



### 1. Find possible lines ###

# connect seperated lines by gradient
    # https://stackoverflow.com/questions/63727525/how-to-connect-broken-lines-that-cannot-be-connected-by-erosion-and-dilation
    # https://stackoverflow.com/questions/43859750/how-to-connect-broken-lines-in-a-binary-image-using-python-opencv

# find all possible lines by graph backtracking
# lines_node : list of lines represented by nodes // ex. [[node0, ..., node3], ..., [node2, ..., node4]]
# temp : list of nodes up to now
# graph : node -> {adj. node -> line between two node} (dictionary type)
# visited_node : visited nodes up to now
# finished_node : visited nodes at least once
# node : current node
def backtrack(lines_node, temp, graph, visited_node, finished_node, node):
    end_pt = True
    for next_node in graph[node].keys():
        if not visited_node[next_node]:
            end_pt = False
            temp.append(next_node)
            visited_node[next_node] = True
            finished_node[next_node] = True
            backtrack(lines_node, temp, graph, visited_node, finished_node, next_node)
            del temp[-1]
            visited_node[next_node] = False
    # if there is no way to preceed, current node is the end node
    # add current line to the list
    if end_pt:
        line_node = []
        line_node.extend(temp)
        lines_node.append(line_node)

# find possible lines
# (1) build a graph
# (2) find all possible lines by graph backtracking
# (3) filter lines with length, direction criteria
def group(img):
    # (1) build a graph
    
    # (1)-1 find all nodes
    count = np.zeros(img.shape)
    nodes = []

    for j in range(1, img.shape[0] - 1):
        for i in range(1, img.shape[1] - 1):
            if img[j, i] == 0: continue
            count[j, i] = np.count_nonzero(img[j-1:j+2, i-1:i+2]) - 1
            if count[j, i] == 1 or count[j, i] >= 3:
                nodes.append((j, i))

    # sort nodes to traverse from upper-left to lower-right
    nodes.sort(key = lambda x : x[0]+x[1])
     
    # (1)-2 save all connections
    # TODO : reduce code length
    graph = dict()
    for node in nodes:
        graph[node] = dict()

    not_visited = np.ones(img.shape)
    for node in nodes:
        y,x = node
        not_visited[y, x] = 0
        around = np.multiply(count[y-1:y+2, x-1:x+2], not_visited[y-1:y+2, x-1:x+2])
        next_pos = np.transpose(np.nonzero(around))
        if next_pos.shape[0] == 0: continue
        for dy,dx in next_pos:
            y,x = node
            next_y = y + dy - 1
            next_x = x + dx - 1
            if dx == 0 or (dy == 0 and dx == 1):
                dy,dx = 2-dy,2-dx
            temp_line = [[y,x,0,0], [next_y,next_x,dy-1,dx-1]]
            if count[next_y, next_x] == 1 or count[next_y, next_x] >= 3:
                not_visited[next_y, next_x] = 1
                graph[tuple(temp_line[0][:2])][tuple(temp_line[-1][:2])] = temp_line
                temp_line_rev = list(reversed(temp_line))
                graph[tuple(temp_line[-1][:2])][tuple(temp_line[0][:2])] = temp_line_rev
                continue
        
            while(True):
                y,x = temp_line[-1][:2]
                not_visited[y, x] = 0
                around = np.multiply(count[y-1:y+2, x-1:x+2], not_visited[y-1:y+2, x-1:x+2])
                next_pos = np.transpose(np.nonzero(around))
                if next_pos.shape[0] == 0: break
                
                # update line
                next_y = y + next_pos[0][0] - 1
                next_x = x + next_pos[0][1] - 1
                dy,dx = next_y-y,next_x-x
                if dx == -1 or (dy == -1 and dx == 0):
                    dy,dx = -dy,-dx
                temp_line.append([next_y, next_x, dy, dx])
                not_visited[next_y, next_x] = 0
                
                # check end condition
                if count[next_y, next_x] == 1 or count[next_y, next_x] >= 3:
                    #if len(temp_line) > 10:
                    graph[tuple(temp_line[0][:2])][tuple(temp_line[-1][:2])] = temp_line
                    temp_line_rev = list(reversed(temp_line))
                    graph[tuple(temp_line[-1][:2])][tuple(temp_line[0][:2])] = temp_line_rev
                    not_visited[next_y, next_x] = 1
                    break
        not_visited[node[0], node[1]] = 1


    # (2) find all possible lines by graph backtracking
    lines_node = []
    visited_node = dict()
    finished_node = dict()
    for node in nodes:
        visited_node[node] = False
        finished_node[node] = False
    
    for node in nodes:
        if not finished_node[node]:
            temp = [node]
            visited_node[node] = True
            finished_node[node] = True
            backtrack(lines_node, temp, graph, visited_node, finished_node, node)
    
    # (3) filter lines with length, direction criteria
    lines = []
    for line_node in lines_node:
        num_node = len(line_node)
        if num_node == 1 : continue
        wrong = False
        line = []
        prev,cur = None,line_node[0]
        for i in range(1,num_node):
            nxt = line_node[i]
            # if the inner product of two connected line segments vectors is <0, discard it
            if i>1 and (cur[0]-prev[0])*(nxt[0]-cur[0])+(cur[1]-prev[1])*(nxt[1]-cur[1])<0:
                wrong = True
                break
            line.extend(graph[cur][nxt])
            prev,cur = cur,nxt
        # if the length is <20, discard it
        if wrong or len(line) < 20: continue
        lines.append(line)
    
    return lines


### 2. Choose three lines ###

# classify lines using l2 distance with centers in feature space
# remain at most 3 lines
def classify(centers, lines):
    classified_lines = [None, None, None]
    line_idx = [None, None, None]
    nearest = [1e9, 1e9, 1e9]
    
    feature_list = np.empty((0,24))
    for line in lines:
        feature = extract_feature(line)
        feature_list = np.vstack((feature_list,feature))
    
    num_lines = len(lines)
    for i in range(3):
        center = centers[i]
        for j in range(num_lines):
            chosen = False
            for k in range(i-1):
                if line_idx[k]==j:
                    chosen = True
                    break
            if chosen: continue
            feature = feature_list[j]
            dist = np.linalg.norm(feature-center)
            if dist < nearest[i]:
                nearest[i] = dist
                classified_lines[i] = lines[j]
                line_idx[i] = j
    
    return classified_lines


### 3. Color each line ###

# color lines with BGR
def color(skel_img, lines):
    color_list = [[255,0,0], [0,255,0], [0,0,255]] # [B,G,R]
    
    num_lines = len(lines)
    colored_img = cv.cvtColor(skel_img,cv.COLOR_GRAY2RGB)
    for i in range(num_lines):
        line = lines[i]
        for y,x,_,_ in line:
            colored_img[y,x] = color_list[i]
            
    return colored_img


### Others ###

# extract feature from a line
def extract_feature(line):
    # feature = [min_y, min_x, max_y, max_x] + mean of direction info(dy,dx) * N intervals
    # => (2N+4)-dim
    feature = np.append(np.min(line, axis=0)[:2], np.max(line, axis=0)[:2])
    N = 10
    step = len(line)//N
    for i in range(N):
        l = line[i*step:(i+1)*step]
        feature = np.append(feature, np.mean(l, axis=0)[2:])
    return feature
      

# find 3 cluster centers in feature space
# we can use pre-trained centers for testing
def get_cluster_centers():
    # prepare good samples
    good = [12,113,220,396,402,429,487,489,493,507,530,777,1089,1100,1150,1207]
    for idx in good:
        rectified = rectify(idx)
        cv.imwrite("good_sample/image"+str(idx)+".png",rectified)
    
    # put all data in feature space
    data = np.empty((0,24))
    for img_path in glob.glob("good_sample/*.png"):
        img = cv.imread(img_path)
        skel_img = cv.cvtColor(skeletonize(img), cv.COLOR_BGR2GRAY)
        lines = group(skel_img)
        for line in lines:
            feature = extract_feature(line)
            data = np.vstack((data,feature))
    
    # k-means clustering (k=3)
    criteria = (cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, centers = cv.kmeans(data.astype(np.float32), 3, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    
    # sort centers according to max_y
    centers = list(centers)
    centers.sort(key = lambda x : x[2])
    
    return centers


if __name__ == "__main__":
    # load (rectified) test data
    num_data = 10
    load_data(num_data)
    
    # find new cluster centers
    #centers = get_cluster_centers()
    
    # trained cluster centers
    centers = [np.array([6.1887500e+02, 7.9262500e+02, 7.6112500e+02, 1.2027500e+03,
                        6.6605198e-01, 8.6089170e-01, 5.4412222e-01, 9.3315840e-01,
                        4.4293469e-01, 1.0000000e+00, 3.4873500e-01, 1.0000000e+00,
                        3.2719675e-01, 1.0000000e+00, 3.1075707e-01, 1.0000000e+00,
                        2.3862401e-01, 1.0000000e+00, 1.9040707e-01, 1.0000000e+00,
                        1.5674138e-01, 1.0000000e+00, 1.5612867e-01, 9.9687499e-01],dtype=np.float32),
               np.array([6.4615002e+02, 5.8690002e+02, 8.4125000e+02, 9.6104999e+02,
                        1.7742984e-01, 9.5737660e-01, 3.1057608e-01, 9.7029686e-01,
                        3.4348270e-01, 1.0000000e+00, 4.2626306e-01, 9.6885043e-01,
                        5.4173779e-01, 9.4824570e-01, 5.6150049e-01, 9.5337659e-01,
                        6.2449318e-01, 9.5198548e-01, 6.6387254e-01, 9.4047660e-01,
                        6.8830168e-01, 9.3963587e-01, 6.5460223e-01, 9.3953258e-01],dtype=np.float32),
               np.array([6.5286957e+02, 5.1626086e+02, 9.2756525e+02, 7.2717395e+02,
                        1.3809928e-01, 9.3377674e-01, 2.7538010e-01, 9.7952598e-01,
                        4.0102530e-01, 9.2006487e-01, 5.5693978e-01, 8.5996735e-01,
                        7.1851170e-01, 7.5060475e-01, 7.2445339e-01, 6.1091334e-01,
                        6.9933695e-01, 4.9578774e-01, 4.2458954e-01, 4.9123678e-01,
                        2.1068455e-01, 5.2427667e-01, 6.1517307e-03, 6.9771338e-01],dtype=np.float32)]
    
    for img_path in glob.glob("line_sample/*.png"):
        print(img_path)
        img = cv.imread(img_path)
        skel_img = cv.cvtColor(skeletonize(img), cv.COLOR_BGR2GRAY)
        lines = group(skel_img)                         # get candidate lines
        classified_lines = classify(centers, lines)     # choose 3 lines from candidates
        colored_img = color(skel_img, classified_lines) # color 3 lines (RGB)
        
        '''
        for i,line in enumerate(lines):
            skel = cv.cvtColor(skel_img,cv.COLOR_GRAY2RGB)
            for y,x,_,_ in line:
                skel[y,x] = [255,0,0]
            cv.imwrite("line_sample_result2/linebyline/line_" + str(i) + "_" + img_path.split(os.sep)[1], skel)
        '''
        
        cv.imwrite("line_sample_result/result_" + img_path.split(os.sep)[1], colored_img)
    
