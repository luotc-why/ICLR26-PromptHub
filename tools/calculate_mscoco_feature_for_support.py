import numpy as np
import scipy.spatial.distance as distance
import os
import sys

current_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if current_path not in sys.path:
    sys.path.append(current_path)
print(sys.path)
import json
import torch
print(torch.cuda.device_count())
print(torch.cuda.is_available())
import models.models_mae as models_mae
from PIL import Image
from evaluate.mae_utils import PURPLE, YELLOW
import torchvision.transforms as T
import h5py
import torchvision.transforms.functional as TF
from PIL import Image
from omegaconf import OmegaConf
from models.vqgan import VQModel

def load_maevq(chkpt_dir = '/data/luotianci/codebase_update/VisualICL/weights/checkpoint-1000.pth',arch='mae_vit_large_patch16'):
    #vq = prepare_model(args.ckpt, arch=args.mae_model)
    # build model
    model = getattr(models_mae, arch)(vq_ckpt_dir='/data/luotianci/codebase_update/VisualICL/weights/vqgan')
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    return model

def extract_ignore_idx(mask, class_id):
    mask = np.array(mask)
    boundary = np.floor(mask / 255.)
    mask[mask != class_id + 1] = 0
    mask[mask == class_id + 1] = 255
    return Image.fromarray(mask), boundary

def read_mask(img_name):
    r"""Return segmentation mask in PIL Image"""
    mask = Image.open(os.path.join(ann_path, img_name)[:-4] + '.png')
    return mask

def read_img(img_name):
    r"""Return RGB image in PIL Image"""
    return Image.open(os.path.join(img_path, img_name)[:-4] + '.jpg')

os.environ["CUDA_VISIBLE_DEVICES"] = '7'

features_name = sys.argv[1]
split = sys.argv[2]

print(f"Processing {features_name} ...")
sys.stdout.flush()

model = load_maevq()
model = model.eval()
model = model.cuda()

datapath = '/data/luotianci/codebase_update/VisualICL/coco'

img_path = os.path.join(datapath, 'trn2014')
ann_path = os.path.join(datapath, 'Coco_Trainlabel')

features_dir = f"/data/luotianci/codebase_update/VisualICL/coco/{features_name}_{split}"
meta_root = f"/data/luotianci/codebase_update/VisualICL/split/coco/{split}"

transform = T.Compose([T.Resize((224, 224), 3),T.ToTensor()])
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).cuda()
for foldid in [0, 1, 2, 3]:
    print(f"Processing folder {foldid}")
    sys.stdout.flush()
    
    with open(os.path.join(meta_root, 'fold'+str(foldid)+'.txt')) as f:
        examples = f.readlines()
    if len(examples) == 0:
        print(f"zeros folder{foldid}")
        sys.stdout.flush()
        continue
    examples = [[data.split('__')[0] , int(data.split('__')[1]) - 1] for data in examples]
        
    imgs = []
    masks = []

    img_global_features = torch.tensor([]).cuda()
    mask_global_features = torch.tensor([]).cuda()
    for k,example in enumerate(examples):
        img_name = example[0]
        class_id = example[1]
        img = transform(read_img(img_name = img_name)).cuda()
        if k%100==0:
            print(k)
        _img = torch.zeros((3,224,224))
        _img[:,:,:] = img
        imgs.append(_img)
        cmask = read_mask(img_name = img_name)
        _mask, ignore_idx = extract_ignore_idx(cmask, class_id=class_id)
        _mask = transform(_mask).cuda()
        mask = torch.zeros((3,224,224))
        mask[:,:,:] = _mask
        masks.append(mask)
        if len(imgs) == 256:
            imgs = torch.stack(imgs).cuda()
            masks = torch.stack(masks).cuda()
            imgs = TF.resize(imgs,(111,111))
            masks = TF.resize(masks,(111,111))
            img_size = 111
            cats = torch.ones(imgs.shape[0],3,224,224).cuda()
            cats[:, :, :img_size, :img_size] = imgs

            cats[:, :, -img_size:, :img_size] = imgs
            cats[:, :, :img_size, -img_size:] = masks
            cats[:, :, -img_size:, -img_size:] = masks
            cats = (cats - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
            # image = TF.to_pil_image(cats[31])
            # print(img_name)
            # # # 保存图像
            # image.save("cat.jpg")
            # print(cats.shape)
            # assert False
            with torch.no_grad():
                img_features = model.patch_embed(cats)
                img_features = img_features + model.pos_embed[:,1:,:]
                if len(img_global_features) == 0:
                    img_global_features = img_features
                else:
                    img_global_features = torch.cat((img_global_features,img_features))
            masks = []
            imgs = []
    if len(imgs)!=0:
        imgs = torch.stack(imgs).cuda()
        masks = torch.stack(masks).cuda()
        imgs = TF.resize(imgs,(111,111))
        masks = TF.resize(masks,(111,111))
        img_size = 111
        cats = torch.ones(imgs.shape[0],3,224,224).cuda()
        cats[:, :, :img_size, :img_size] = imgs

        cats[:, :, -img_size:, :img_size] = imgs
        cats[:, :, :img_size, -img_size:] = masks
        cats[:, :, -img_size:, -img_size:] = masks
        cats = (cats - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
        with torch.no_grad():
            img_features = model.patch_embed(cats)
            img_features = img_features + model.pos_embed[:,1:,:]
            if len(img_global_features) == 0:
                img_global_features = img_features
            else:
                img_global_features = torch.cat((img_global_features,img_features))
        
    img_features = img_global_features.cpu().numpy().astype(np.float32)
    with h5py.File(f"{features_dir}/folder{foldid}_features_by_vqgan_encoder.h5df", "w") as f:
        for i in range(len(examples)):
            dataset_name = examples[i][0]
            print(i,"   ",dataset_name)
            if dataset_name not in f:
                feature = img_features[i]
                # print(feature.shape)
                # print(feature.shape,(feature[:98,:]).shape)
                dset = f.create_dataset(dataset_name, data = feature[:98,:])