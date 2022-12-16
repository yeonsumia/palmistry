# Fortune On Your Hand: View-Invariant Machine Palmistry

## Environment
The codes are written based on Python 3.7.6. These are the requirements for running the codes:
- torch
- torchvision
- scikit-image
- opencv-python
- pillow-heif
- mediapipe

In order to install the requirements, run `pip install -r requirements.txt`.

## How to run
1. Before running the codes, **a palm image for input(.heic or .jpg)** should be prepared in the `inputs` directory. We provided four sample inputs.
2. Run `read_palm.py` by the command below. After running the code, result files will be saved in the `results` directory.
```bash
$> python read_palm.py --input [input_file_name.jpg]
```