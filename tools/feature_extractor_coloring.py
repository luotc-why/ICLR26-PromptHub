"""
Extract features for SupPR.
"""
import os
import sys
import glob
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
import torchvision.models as models
from torchvision import transforms as T
from torch.nn import functional as F

import timm
from timm.models import load_checkpoint
from collections import OrderedDict

model_name = sys.argv[1]
feature_name = sys.argv[2]
split = sys.argv[3]

pretrained_cfg = timm.models.create_model('vit_large_patch14_clip_224').default_cfg
pretrained_cfg['file'] ='/data/luotianci/visual_prompt_retrieval/tools/vit_large_patch14_clip_224.laion2b/open_clip_pytorch_model.bin'
model = timm.create_model(model_name, pretrained=True,pretrained_cfg=pretrained_cfg)
model.eval()
model = model.cuda()

# load the image transformer
t = []
size = 224
t.append(T.Resize((size,size), interpolation=Image.BICUBIC))
t.append(T.CenterCrop(size))
t.append(T.ToTensor())
t.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
center_crop = T.Compose(t)

save_dir = f"./imagenet/{feature_name}_{split}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    print(f"Directory exists at {save_dir}")
    sys.exit()

if split == 'trn':
    image_root = "./imagenet/train_data"
else :
    image_root = "./imagenet/test_data"

examples = [os.path.join(image_root, file) for file in os.listdir(image_root)]
    
imgs = []

global_features = torch.tensor([]).cuda()
for example in examples:
    try:
        path = os.path.join(example)
        img = Image.open(path).convert("RGB")
        img = center_crop(img)
        imgs.append(img)
    except:
        print(f"Disappear {path}")
        sys.stdout.flush()

    if len(imgs) == 128:

        imgs = torch.stack(imgs).cuda()
        with torch.no_grad():
            features = model(imgs)
            if len(global_features) == 0:
                global_features = features
            else:
                global_features = torch.cat((global_features,features))

        imgs = []

imgs = torch.stack(imgs).cuda()
with torch.no_grad():
    features = model(imgs)
    if len(global_features) == 0:
        global_features = features
    else:
        global_features = torch.cat((global_features,features))

features = global_features.cpu().numpy().astype(np.float32)

save_file = os.path.join(save_dir)
np.savez(save_file, examples=examples, features=features)
