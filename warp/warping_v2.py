import glob
import cv2 as cv
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import warping_v1

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
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

# For static images:
IMAGE_FILES = [img_path for img_path in glob.glob('./data/*')]
IMAGE_FILES = [IMAGE_FILES[i] for i in [3,6,9,12,15,19,20]]
with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # 1. Extract 21 landmark points
    # Read an image, flip it around y-axis for correct handedness output
    image = cv.flip(cv.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv.imwrite('./results2/annotated_image' + str(idx) + '.png', cv.flip(annotated_image, 1))
    
    # 2. Align images
    pts = np.float32([[hand_landmarks.landmark[i].x*image_width,
                       hand_landmarks.landmark[i].y*image_height] for i in pts_index])
    pts_target = np.float32([[x*image_width, y*image_height] for x,y in pts_target_normalized])
    M, mask = cv.findHomography(pts, pts_target, cv.RANSAC,5.0)
    warped_image = cv.warpPerspective(annotated_image, M, (image_width, image_height))
    cv.imwrite('./results2/warped_image' + str(idx) + '.png', warped_image)
    