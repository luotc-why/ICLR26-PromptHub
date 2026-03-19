"""
Extract features for UnsupPR.
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
import sys
from evaluate_detection.voc_orig import VOCDetection4Val, VOCDetection4Train, make_transforms

model_name = sys.argv[1]
feature_name = sys.argv[2]

"""
model names:
# vit_large_patch14_224_clip_laion2b
# eva_large_patch14_196.in22k_ft_in22k_in1k
# resnet50
# vit_large_patch16_224.augreg_in21k_ft_in1k
# resnet18
# vit_large_patch14_clip_224.laion2b_ft_in12k_in1k
# vit_base_patch16_224.dino
"""
pretrained_cfg = timm.models.create_model('vit_large_patch14_clip_224').default_cfg
pretrained_cfg['file'] ='/data/luotianci/visual_prompt_retrieval/tools/vit_large_patch14_clip_224.laion2b/open_clip_pytorch_model.bin'
model = timm.create_model(model_name, pretrained=True,pretrained_cfg=pretrained_cfg)
model.eval()
model = model.cuda()

# load the image transformer
t = []
t.append(T.Resize(model.pretrained_cfg['input_size'][1], interpolation=Image.BICUBIC))
t.append(T.CenterCrop(model.pretrained_cfg['input_size'][1]))
t.append(T.ToTensor())
t.append(T.Normalize(model.pretrained_cfg['mean'], model.pretrained_cfg['std']))
center_crop = T.Compose(t)
pascal_path='pascal-5i'
years=("2012",)

eval_support = VOCDetection4Val(pascal_path, years, image_sets=['train'], transforms=None,keep_single_objs_only=1, filter_by_mask_size=1)
train_support = VOCDetection4Train(pascal_path, years, image_sets=['train'], transforms=None, keep_single_objs_only=1, filter_by_mask_size=1)

eval_query = VOCDetection4Val(pascal_path, years, image_sets=['val'], transforms=None,keep_single_objs_only=1, filter_by_mask_size=1)
train_query = VOCDetection4Train(pascal_path, years, image_sets=['val'], transforms=None, keep_single_objs_only=1, filter_by_mask_size=1)


image_root = "./pascal-5i/VOC2012/JPEGImages"
sys.stdout.flush()

save_dir = f"./pascal-5i/VOC2012/{feature_name}_val_all_detection"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
examples = []
for x in eval_support.images:
    examples.append(x.split('/')[-1][:-4])

if len(examples) == 0:
    print(f"zeros file.")
    sys.stdout.flush()

examples = [os.path.join(image_root, example.strip()+'.jpg') for example in examples]

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
            features = model.forward_features(imgs)
            if len(global_features) == 0:
                global_features = features
            else:
                global_features = torch.cat((global_features, features))

        imgs = []

imgs = torch.stack(imgs).cuda()
with torch.no_grad():
    features = model.forward_features(imgs)
    if len(global_features) == 0:
        global_features = features
    else:
        global_features = torch.cat((global_features, features))

features = global_features.cpu().numpy().astype(np.float32)

save_file = os.path.join(save_dir, 'folder_support')
np.savez(save_file, examples=examples, features=features)

print('features shape: ', features.shape)

examples = []
for x in eval_query.images:
    examples.append(x.split('/')[-1][:-4])

if len(examples) == 0:
    print(f"zeros file.")
    sys.stdout.flush()

examples = [os.path.join(image_root, example.strip()+'.jpg') for example in examples]

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
            features = model.forward_features(imgs)
            if len(global_features) == 0:
                global_features = features
            else:
                global_features = torch.cat((global_features, features))

        imgs = []

imgs = torch.stack(imgs).cuda()
with torch.no_grad():
    features = model.forward_features(imgs)
    if len(global_features) == 0:
        global_features = features
    else:
        global_features = torch.cat((global_features, features))

features = global_features.cpu().numpy().astype(np.float32)

save_file = os.path.join(save_dir, 'folder_query')
np.savez(save_file, examples=examples, features=features)

print('features shape: ', features.shape)

save_dir = f"./pascal-5i/VOC2012/{feature_name}_train_all_detection"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

examples = []
for x in train_support.images:
    examples.append(x.split('/')[-1][:-4])

if len(examples) == 0:
    print(f"zeros file.")
    sys.stdout.flush()

examples = [os.path.join(image_root, example.strip()+'.jpg') for example in examples]
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
            features = model.forward_features(imgs)
            if len(global_features) == 0:
                global_features = features
            else:
                global_features = torch.cat((global_features, features))

        imgs = []

imgs = torch.stack(imgs).cuda()
with torch.no_grad():
    features = model.forward_features(imgs)
    if len(global_features) == 0:
        global_features = features
    else:
        global_features = torch.cat((global_features, features))

features = global_features.cpu().numpy().astype(np.float32)

save_file = os.path.join(save_dir, 'folder_support')
np.savez(save_file, examples=examples, features=features)

print('features shape: ', features.shape)

examples = []
for x in train_query.images:
    examples.append(x.split('/')[-1][:-4])

if len(examples) == 0:
    print(f"zeros file.")
    sys.stdout.flush()

examples = [os.path.join(image_root, example.strip()+'.jpg') for example in examples]

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
            features = model.forward_features(imgs)
            if len(global_features) == 0:
                global_features = features
            else:
                global_features = torch.cat((global_features, features))

        imgs = []

imgs = torch.stack(imgs).cuda()
with torch.no_grad():
    features = model.forward_features(imgs)
    if len(global_features) == 0:
        global_features = features
    else:
        global_features = torch.cat((global_features, features))

features = global_features.cpu().numpy().astype(np.float32)

save_file = os.path.join(save_dir, 'folder_query')
np.savez(save_file, examples=examples, features=features)

print('features shape: ', features.shape)
