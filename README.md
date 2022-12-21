# Fortune On Your Hand: View-Invariant Machine Palmistry
## Summary
Our *Palmistry principal lines detection software* is implemented by 4 steps below. Our main challenge was to read the principal lines on a palm regardless of the **view direction** and **illumination**:   
1) Warping a tilted palm image  
2) Detecting principal lines on a palm  
3) Classifying the lines  
4) Measuring the length of each line  
<img width="1362" alt="model_architecture" src="https://user-images.githubusercontent.com/81272473/208795260-48ba6c8f-92a1-4b01-9471-6a4703ad0aff.png">
For palm image rectification, we used MediaPipe to extract interest points and implemented warping with the points. For principal line detection, we built a deep learning model and trained the model with palm image dataset. For line classification, we used K-means clustering to allocate each pixel to specific line. For length measurement, we set a threshold for each principal line with the landmarks obtained by MediaPipe.

## Environment
The codes are written based on Python 3.7.6. These are the requirements for running the codes:
- torch
- torchvision
- scikit-image
- opencv-python
- pillow-heif
- mediapipe

In order to install the requirements, run `pip install -r ./code/requirements.txt`.

## Run
1. Before running the codes, **a palm image for input(.heic or .jpg)** should be prepared in the `./code/inputs` directory. We provided four sample inputs.
2. Run `read_palm.py` by the command below. After running the code, result files will be saved in the `./code/results` directory.
```bash
> python ./code/read_palm.py --input [filename].[jpg, heic]
```

## Results
<img width="1371" alt="standard" src="https://user-images.githubusercontent.com/81272473/208797334-9cf56f18-01b1-46e5-9bab-5a38a696d05f.png">
<img width="1361" alt="tilted" src="https://user-images.githubusercontent.com/81272473/208797357-fe007daf-0d24-48b0-80af-21d79b64db4a.png">

## Line Segment implementation
Update: 22.12.03 21:57
- Assumption
  - line이 image의 테두리까지 가는 경우가 없음 (이 경우 scikit의 skeletonize가 종종 안됨. skeletonize 되더라도 grouping 알고리즘 조금 수정 필요)
  - 선들이 교차하는 점은 최대 하나 (test case에 따랐음. 약간의 추가 구현으로 처리 가능하기는 함)

- line grouping
  - return value : list of lines, each lines are also a list of pixels
  ```
  example : [ [[1, 2], [2, 3]], [[10, 11], [11, 11]] ]
  ```
  
  - explanation of implementation
    1. 전체 픽셀에 대해 둘레 8픽셀 중 0이 아닌 값을 count
    2. count 결과물은 0: 선 위에 없음, 1: 선의 끝, 2: 선의 중간, 3: 선의 교차점으로 구분됨
    3. 선의 끝인 pixel에서 시작해서 주변 8픽셀을 탐색, count가 0이 아니고 방문하지 않은 pixel을 따라감
    4. 가다보면 count가 1이나 3인 pixel에 도달
    5. 1인 pixel이면 line을 하나 찾은 것이므로 저장하고 역방향 탐색이 되지 않도록 for문에서 제외. 3인 pixel은 line을 따로 저장해놨다가 추후 조치
    6. 3으로 끝난 line들끼리 이을 수 있나 확인: 시작점, 끝점 차이 확인해서 방향이 반대인 모든 조합들을 이어서 line에 저장
    7. 저장한 line들을 return
    
## Issues
  - skeletonize가 붙어있지 않던 선을 붙이는 경우 있음 (1 case, 선 하나가 약간 길게 나오게 됨) -> 추가 test 필요
  - 끊어진 라인 처리가 애매함 : 현재는 무시하고 진행한 상태, grouping된 선들 gradient 계산하면 할 수야 있기는 한데 잘못하면 이상한 선들끼리 이어질 수 있음. 이런 케이스를 숨기는게 좋아보이긴 함...
