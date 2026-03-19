import numpy as np
import scipy.spatial.distance as distance
import os
import sys
import json
import torch
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cwd))
from models import models_mae
from PIL import Image
from evaluate.mae_utils import PURPLE, YELLOW
import torchvision.transforms as T
import h5py
import torchvision.transforms.functional as TF
from PIL import Image
from omegaconf import OmegaConf
from models.vqgan import VQModel

def load_maevq(chkpt_dir = './weights/checkpoint-1000.pth',arch='mae_vit_large_patch16'):
    #vq = prepare_model(args.ckpt, arch=args.mae_model)
    # build model
    model = getattr(models_mae, arch)()
    # load model
    full_path = os.path.join(basedir, chkpt_dir)
    checkpoint = torch.load(full_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    return model


def read_mask(img_name):
    r"""Return segmentation mask in PIL Image"""
    mask = Image.open(os.path.join(ann_path, img_name))
    return mask

def read_img(img_name):
    r"""Return RGB image in PIL Image"""
    return Image.open(os.path.join(img_path, img_name))

features_name = sys.argv[1]
split = sys.argv[2]
cudaid = sys.argv[3]
basedir = sys.argv[4]

print(f"Processing {features_name} ...")
sys.stdout.flush()

model = load_maevq()
model = model.eval()
model = model.to(cudaid)


if split == 'trn':
    img_path = './imagenet/train_data'
    ann_path = './imagenet/train_label'
else :
    img_path = './imagenet/test_data'
    ann_path = './imagenet/test_label'
img_path = os.path.join(basedir, img_path)
ann_path = os.path.join(basedir, ann_path)
transform = T.Compose([
    T.Lambda(lambda img: img.convert('RGB')),  
    T.Resize((224, 224)), 
    T.ToTensor(),  
])
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).to(cudaid)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).to(cudaid)

if split == 'trn':
    image_root = "./imagenet/train_data"
else :
    image_root = "./imagenet/test_data"
image_root = os.path.join(basedir, image_root)

features_dir = f"./imagenet/{features_name}_{split}"
features_dir = os.path.join(basedir, features_dir)

examples = [os.path.join(image_root, file) for file in os.listdir(image_root)]

if len(examples) == 0:
    print(f"zeros folder ...")
    sys.stdout.flush()

        
imgs = []
masks = []

img_global_features = torch.tensor([]).to('cpu')
for k,example in enumerate(examples):
    image_path = example
    img_name = image_path.strip().split('/')[-1]
    _img = transform(read_mask(img_name = img_name)).to(cudaid)
    img = torch.zeros((3,224,224))
    img[:,:,:] = _img
    if k%100==0:
        print(k)
    cmask = read_img(img_name = img_name)
    _mask = transform(cmask).to(cudaid)
    mask = torch.zeros((3,224,224))
    mask[:,:,:] = _mask
    masks.append(img)
    imgs.append(mask)
    if len(imgs) == 256:
        imgs = torch.stack(imgs).to(cudaid)
        masks = torch.stack(masks).to(cudaid)
        imgs = TF.resize(imgs,(111,111))
        masks = TF.resize(masks,(111,111))
        img_size = 111
        cats = torch.ones(imgs.shape[0],3,224,224).to(cudaid)
        cats[:, :, :img_size, :img_size] = imgs

        cats[:, :, -img_size:, :img_size] = imgs
        cats[:, :, :img_size, -img_size:] = masks
        cats[:, :, -img_size:, -img_size:] = masks
        cats = (cats - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
        with torch.no_grad():
            img_features = model.patch_embed(cats)
            img_features = img_features + model.pos_embed[:,1:,:]
            img_features = img_features.to('cpu')
            if len(img_global_features) == 0:
                img_global_features = img_features
            else:
                img_global_features = torch.cat((img_global_features,img_features))
        masks = []
        imgs = []
if len(imgs)!=0:
    imgs = torch.stack(imgs).to(cudaid)
    masks = torch.stack(masks).to(cudaid)
    imgs = TF.resize(imgs,(111,111))
    masks = TF.resize(masks,(111,111))
    img_size = 111
    cats = torch.ones(imgs.shape[0],3,224,224).to(cudaid)
    cats[:, :, :img_size, :img_size] = imgs

    cats[:, :, -img_size:, :img_size] = imgs
    cats[:, :, :img_size, -img_size:] = masks
    cats[:, :, -img_size:, -img_size:] = masks
    cats = (cats - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
    with torch.no_grad():
        img_features = model.patch_embed(cats)
        img_features = img_features + model.pos_embed[:,1:,:]
        img_features = img_features.to('cpu')
        if len(img_global_features) == 0:
            img_global_features = img_features
        else:
            img_global_features = torch.cat((img_global_features,img_features))
    
img_features = img_global_features.cpu().numpy().astype(np.float32)
with h5py.File(f"{features_dir}/folder_query_features_by_vqgan_encoder.h5df", "w") as f:
    for i in range(len(examples)):
        if split == 'trn':
            dataset_name = examples[i].strip().split('/')[-1][:-4]
        else :
            dataset_name = examples[i].strip().split('/')[-1][:-5]
        print(i,"   ",dataset_name)
        if dataset_name not in f:
            feature = img_features[i]
            dset = f.create_dataset(dataset_name, data = feature[98:,:])