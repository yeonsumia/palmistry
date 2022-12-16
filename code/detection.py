import numpy as np
from PIL import Image
import torch

def detect(net, jpeg_dir, output_dir, resize_value, device=torch.device('cpu')):
    pil_img = Image.open(jpeg_dir)
    img = np.asarray(pil_img.resize((resize_value, resize_value), resample=Image.NEAREST)) / 255
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).permute(0,3,1,2).to(device)
    pred = net(img).squeeze(0)
    pred = torch.Tensor(np.apply_along_axis(lambda x: [1,1,1] if x > 0.03 else [0,0,0], 0, pred.cpu().detach()))
    Image.fromarray((pred.permute((1,2,0)).numpy() * 255).astype(np.uint8)).save(output_dir)