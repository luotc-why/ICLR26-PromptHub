import numpy as np
import scipy.spatial.distance as distance
import os
import sys
import json
import cv2
import sys
from evaluate_detection.voc_orig import VOCDetection4Val, VOCDetection4Train, make_transforms
from evaluate_detection.voc import make_transforms, create_grid_from_images
import torch
import models.models_mae as models_mae
from PIL import Image
from evaluate.mae_utils import PURPLE, YELLOW
import torchvision.transforms as T
import h5py
import torchvision.transforms.functional as TF
from PIL import Image
from omegaconf import OmegaConf
from models.vqgan import VQModel

def box_to_img(mask, target, border_width=4):
    if mask is None:
        mask = np.zeros((112, 112, 3))
    h, w, _ = mask.shape
    for box in target['boxes']:
        x_min, y_min, x_max, y_max = list((box * (h - 1)).round().int().numpy())
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), (255, 255, 255), border_width)
    return Image.fromarray(mask.astype('uint8'))


def get_annotated_image(img, boxes, border_width=3, mode='draw', bgcolor='white', fg='image'):
    if mode == 'draw':
        image_copy = np.array(img.copy())
        for box in boxes:
            box = box.numpy().astype('int')
            cv2.rectangle(image_copy, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), border_width)
    elif mode == 'keep':
        image_copy = np.array(Image.new('RGB', (img.shape[1], img.shape[0]), color=bgcolor))

        for box in boxes:
            box = box.numpy().astype('int')
            if fg == 'image':
                image_copy[box[1]:box[3], box[0]:box[2]] = img[box[1]:box[3], box[0]:box[2]]
            elif fg == 'white':
                image_copy[box[1]:box[3], box[0]:box[2]] = 255

    return image_copy

def load_maevq(chkpt_dir = './weights/checkpoint-1000.pth',arch='mae_vit_large_patch16'):
    #vq = prepare_model(args.ckpt, arch=args.mae_model)
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    return model

features_name = sys.argv[1]

print(features_name)

print(f"Processing {features_name} ...")
sys.stdout.flush()

model = load_maevq()
model = model.eval()
model = model.cuda()
features_dir = f"./pascal-5i/VOC2012/{features_name}_support_all_detection"
if not os.path.exists(features_dir):
    os.makedirs(features_dir)
pascal_path='pascal-5i'
years=("2012",)

eval_support = VOCDetection4Val(pascal_path, years, image_sets=['train'], transforms=None,keep_single_objs_only=1, filter_by_mask_size=1)
train_support = VOCDetection4Train(pascal_path, years, image_sets=['train'], transforms=None, keep_single_objs_only=1, filter_by_mask_size=1)


background_transforms = T.Compose([
    T.Resize((224, 224)),
    T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
])
transforms = make_transforms('val')

print(len(eval_support.images))
print(len(train_support.images))

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

img_global_features = torch.tensor([]).cuda()

cats = []

for idx in range(len(eval_support.images)):    
    query_image, query_target = eval_support[idx]

    label = query_target['labels'].numpy()[0]

    boxes = query_target['boxes'][torch.where(query_target['labels'] == label)[0]]
    query_image_copy = get_annotated_image(np.array(query_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
    query_image_copy_pil = Image.fromarray(query_image_copy)

    query_image_ten = transforms(query_image, None)[0]
    query_target_ten = transforms(query_image_copy_pil, None)[0]

    background_image = Image.new('RGB', (224, 224), color='white')
    background_image = background_transforms(background_image)
    grid = create_grid_from_images(background_image, query_image_ten, query_target_ten, query_image_ten,
                                    query_target_ten)
    
    cats.append(grid)
    if len(cats) == 256:
        cats = torch.stack(cats).cuda()
        print(cats.shape)
        with torch.no_grad():
            img_features = model.patch_embed(cats)
            img_features = img_features + model.pos_embed[:,1:,:]
            if len(img_global_features) == 0:
                img_global_features = img_features
            else:
                img_global_features = torch.cat((img_global_features,img_features))
        cats = []

if len(cats)!=0:
    with torch.no_grad():
        cats = torch.stack(cats).cuda()
        img_features = model.patch_embed(cats)
        img_features = img_features + model.pos_embed[:,1:,:]
        if len(img_global_features) == 0:
            img_global_features = img_features
        else:
            img_global_features = torch.cat((img_global_features,img_features))
    cats = []
img_features = img_global_features.cpu().numpy().astype(np.float32)

with h5py.File(f"{features_dir}/detection_eval_support.h5df", "w") as f:
    for idx in range(len(eval_support.images)):
        query_image_name = eval_support.images[idx].split('/')[-1][:-4]    
        dataset_name = query_image_name
        print(idx,"   ",dataset_name)
        if dataset_name not in f:
            feature = img_features[idx]
            dset = f.create_dataset(dataset_name, data = feature[:98,:])

img_global_features = torch.tensor([]).cuda()

cats = []

for idx in range(len(train_support.images)):    
    query_image, query_target = train_support[idx]

    label = query_target['labels'].numpy()[0]

    boxes = query_target['boxes'][torch.where(query_target['labels'] == label)[0]]
    query_image_copy = get_annotated_image(np.array(query_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
    query_image_copy_pil = Image.fromarray(query_image_copy)

    query_image_ten = transforms(query_image, None)[0]
    query_target_ten = transforms(query_image_copy_pil, None)[0]

    background_image = Image.new('RGB', (224, 224), color='white')
    background_image = background_transforms(background_image)
    grid = create_grid_from_images(background_image, query_image_ten, query_target_ten, query_image_ten,
                                    query_target_ten)
    
    cats.append(grid)
    if len(cats) == 256:
        cats = torch.stack(cats).cuda()
        with torch.no_grad():
            img_features = model.patch_embed(cats)
            img_features = img_features + model.pos_embed[:,1:,:]
            if len(img_global_features) == 0:
                img_global_features = img_features
            else:
                img_global_features = torch.cat((img_global_features,img_features))
        cats = []

if len(cats)!=0:
    with torch.no_grad():
        cats = torch.stack(cats).cuda()
        img_features = model.patch_embed(cats)
        img_features = img_features + model.pos_embed[:,1:,:]
        if len(img_global_features) == 0:
            img_global_features = img_features
        else:
            img_global_features = torch.cat((img_global_features,img_features))
    cats = []
img_features = img_global_features.cpu().numpy().astype(np.float32)

with h5py.File(f"{features_dir}/detection_train_support.h5df", "w") as f:
    for idx in range(len(train_support.images)):
        query_image_name = train_support.images[idx].split('/')[-1][:-4]    
        dataset_name = query_image_name
        print(idx,"   ",dataset_name)
        if dataset_name not in f:
            feature = img_features[idx]
            dset = f.create_dataset(dataset_name, data = feature[:98,:])