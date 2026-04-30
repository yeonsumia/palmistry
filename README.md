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
