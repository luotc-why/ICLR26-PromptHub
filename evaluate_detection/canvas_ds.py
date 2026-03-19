import torch.utils.data as data
import sys
import os
import argparse

relative_path = './evaluate_detection'
abs_path = os.path.abspath(relative_path)

sys.path.append(abs_path)

from .voc_orig import VOCDetection4Val, VOCDetection4Train, make_transforms
import cv2
from PIL import Image
from .voc import make_transforms, create_grid_from_images
import torch
import numpy as np
import torchvision.transforms as T
import json
import torchvision.transforms.functional as TTTT
import h5py


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


class CanvasDataset4Val(data.Dataset):
    def __init__(self, pascal_path='pascal-5i',args = None, years=("2012",),simidx = 1, random=False, **kwargs):
        self.train_ds = VOCDetection4Val(pascal_path, years, image_sets=['train'], transforms=None,
                                         keep_single_objs_only=1, filter_by_mask_size=1)
        self.val_ds = VOCDetection4Val(pascal_path, years, image_sets=['val'], transforms=None,
                                       keep_single_objs_only=1, filter_by_mask_size=1)
        self.pascal_pat = pascal_path
        self.background_transforms = T.Compose([
            T.Resize((224, 224)),
            T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ])
        self.transforms = make_transforms('val')
        self.random = random
        self.simidx = simidx
        self.images_top50 = self.get_top50_images()
        self.img_feature_for_train_path = './VOC2012/features_vit-laion2b_pixel-level_query_all_detection/detection_eval_query.h5df'
        self.img_feature_for_train_path = os.path.join(pascal_path, self.img_feature_for_train_path)
        self.support_feature_for_train_path = './VOC2012/features_vit-laion2b_pixel-level_support_all_detection/detection_eval_support.h5df'
        self.support_feature_for_train_path = os.path.join(pascal_path, self.support_feature_for_train_path)
        self.args = args

    def get_top50_images(self):
        with open(f'{self.pascal_pat}/VOC2012/features_vit-laion2b_pixel-level_val_all_detection/new_top_50-similarity.json') as f:
            images_top50 = json.load(f)
        return images_top50
    
    def __len__(self):
        return len(self.val_ds)
    
    def __getitem__(self, idx):
        grids = torch.tensor([]) 
        support_imgs = torch.tensor([]) 
        support_masks = torch.tensor([]) 
        query_img_features = torch.tensor([]) 
        support_features = torch.tensor([]) 
        for sim_idx in range(self.simidx):
            if sim_idx == 0:
                query_image, query_target = self.val_ds[idx]

                query_image_name = self.val_ds.images[idx].split('/')[-1][:-4]
                label = query_target['labels'].numpy()[0]
                support_image_name = self.images_top50[query_image_name][sim_idx]
            

                support_image, support_target = self.train_ds[self.train_ds.nameToNum[support_image_name]]
                support_label = support_target['labels'].numpy()[0]
                boxes = support_target['boxes'][torch.where(support_target['labels'] == support_label)[0]]
                support_image_copy = get_annotated_image(np.array(support_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
                support_image_copy_pil = Image.fromarray(support_image_copy)

                boxes = query_target['boxes'][torch.where(query_target['labels'] == label)[0]]
                query_image_copy = get_annotated_image(np.array(query_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
                query_image_copy_pil = Image.fromarray(query_image_copy)

                query_image_ten = self.transforms(query_image, None)[0]
                query_target_ten = self.transforms(query_image_copy_pil, None)[0]
                support_target_ten = self.transforms(support_image_copy_pil, None)[0]
                support_image_ten = self.transforms(support_image, None)[0]

                background_image = Image.new('RGB', (224, 224), color='white')
                background_image = self.background_transforms(background_image)
                grid = create_grid_from_images(background_image, support_image_ten, support_target_ten, query_image_ten,
                                                query_target_ten)
                query_feature, support_feature = self.load_feature(query_image_name,support_image_name)

                query_img_features = torch.tensor(query_feature).unsqueeze(0)
                support_features = torch.tensor(support_feature).unsqueeze(0)
                support_imgs = support_image_ten.unsqueeze(0)
                support_masks = support_target_ten.unsqueeze(0)
                grids = grid.unsqueeze(0)

            else:
                support_image_name = self.images_top50[query_image_name][sim_idx]

                support_image, support_target = self.train_ds[self.train_ds.nameToNum[support_image_name]]
                support_label = support_target['labels'].numpy()[0]
                # print(query_image_name,support_image_name)
                boxes = support_target['boxes'][torch.where(support_target['labels'] == support_label)[0]]
                support_image_copy = get_annotated_image(np.array(support_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
                support_image_copy_pil = Image.fromarray(support_image_copy)

                boxes = query_target['boxes'][torch.where(query_target['labels'] == label)[0]]
                query_image_copy = get_annotated_image(np.array(query_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
                query_image_copy_pil = Image.fromarray(query_image_copy)

                query_image_ten = self.transforms(query_image, None)[0]
                query_target_ten = self.transforms(query_image_copy_pil, None)[0]
                support_target_ten = self.transforms(support_image_copy_pil, None)[0]
                support_image_ten = self.transforms(support_image, None)[0]

                background_image = Image.new('RGB', (224, 224), color='white')
                background_image = self.background_transforms(background_image)
                grid = create_grid_from_images(background_image, support_image_ten, support_target_ten, query_image_ten,
                                                query_target_ten)
                support_img = support_image_ten.unsqueeze(0)
                support_mask = support_target_ten.unsqueeze(0)
                grid = grid.unsqueeze(0)
                _,  support_feature = self.load_feature(query_image_name,support_image_name)
                support_feature = torch.tensor(support_feature).unsqueeze(0)
                support_features = torch.cat((support_features,support_feature))
                support_imgs = torch.cat((support_imgs,support_img))
                support_masks = torch.cat((support_masks,support_mask))
                grids = torch.cat((grids,grid))

        batch = {'query_img': query_image_ten,
            'query_mask': query_target_ten,
            'support_imgs': support_imgs,
            'support_masks': support_masks,
            'grids': grids,
            'query_img_features': query_img_features,
            'support_features': support_features
        }
        return batch
    def load_feature(self,query_name, support_name):
        with h5py.File(self.img_feature_for_train_path, "r") as f:
            query_img_feature = f[query_name][...]
        with h5py.File(self.support_feature_for_train_path, "r") as f:
            support_feature = f[support_name][...]
        return query_img_feature,support_feature

class CanvasDataset4Train(data.Dataset):
    def __init__(self, pascal_path='pascal-5i', args = None,years=("2012",), simidx = 1, random=False, **kwargs):
        self.train_ds = VOCDetection4Train(pascal_path, years, image_sets=['train'], transforms=None, keep_single_objs_only=1, filter_by_mask_size=1)
        self.val_ds = VOCDetection4Train(pascal_path, years, image_sets=['val'], transforms=None, keep_single_objs_only=1, filter_by_mask_size=1)
        self.pascal_pat = pascal_path
        self.background_transforms = T.Compose([
            T.Resize((224, 224)),
            T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ])
        self.transforms = make_transforms('val')
        self.random = random
        self.simidx = simidx
        self.images_top50 = self.get_top50_images()
        self.img_feature_for_train_path = './VOC2012/features_vit-laion2b_pixel-level_query_all_detection/detection_train_query.h5df'
        self.img_feature_for_train_path = os.path.join(pascal_path, self.img_feature_for_train_path)
        self.support_feature_for_train_path = './VOC2012/features_vit-laion2b_pixel-level_support_all_detection/detection_train_support.h5df'
        self.support_feature_for_train_path = os.path.join(pascal_path, self.support_feature_for_train_path)
        self.args = args
    def get_top50_images(self):
        with open(f'{self.pascal_pat}/VOC2012/features_vit-laion2b_pixel-level_train_all_detection/new_top_50-similarity.json') as f:
            images_top50 = json.load(f)
        return images_top50
    
    def __len__(self):
        return len(self.val_ds)
    
    def load_feature(self,query_name, support_name):
        with h5py.File(self.img_feature_for_train_path, "r") as f:
            query_img_feature = f[query_name][...]
        with h5py.File(self.support_feature_for_train_path, "r") as f:
            support_feature = f[support_name][...]
        return query_img_feature,support_feature
    
    def __getitem__(self, idx):
        grids = torch.tensor([]) 
        support_imgs = torch.tensor([]) 
        support_masks = torch.tensor([]) 
        query_img_features = torch.tensor([]) 
        support_features = torch.tensor([]) 
        for sim_idx in range(self.simidx):
            if sim_idx == 0:
                query_image, query_target = self.val_ds[idx]
                query_image_name = self.val_ds.images[idx].split('/')[-1][:-4]
                label = query_target['labels'].numpy()[0]
                rand_value = torch.rand(1).item()
                if rand_value < self.args.pq:
                    support_image, support_target = query_image, query_target
                    support_image_name = query_image_name
                elif self.args.pq <= rand_value < self.args.pq + self.args.pr:
                    random_idx = torch.randint(len(self.train_ds), (1,)).item()
                    support_image, support_target = self.train_ds[random_idx]
                    support_image_name = self.train_ds.images[random_idx].split('/')[-1][:-4]
                else:
                    support_image_name = self.images_top50[query_image_name][sim_idx]
                    support_image, support_target = self.train_ds[self.train_ds.nameToNum[support_image_name]]
                support_label = support_target['labels'].numpy()[0]
                boxes = support_target['boxes'][torch.where(support_target['labels'] == support_label)[0]]
                support_image_copy = get_annotated_image(np.array(support_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
                support_image_copy_pil = Image.fromarray(support_image_copy)

                boxes = query_target['boxes'][torch.where(query_target['labels'] == label)[0]]
                query_image_copy = get_annotated_image(np.array(query_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
                query_image_copy_pil = Image.fromarray(query_image_copy)

                query_image_ten = self.transforms(query_image, None)[0]
                query_target_ten = self.transforms(query_image_copy_pil, None)[0]
                support_target_ten = self.transforms(support_image_copy_pil, None)[0]
                support_image_ten = self.transforms(support_image, None)[0]

                background_image = Image.new('RGB', (224, 224), color='white')
                background_image = self.background_transforms(background_image)
                grid = create_grid_from_images(background_image, support_image_ten, support_target_ten, query_image_ten,
                                                query_target_ten)
                query_feature, support_feature = self.load_feature(query_image_name,support_image_name)
                query_img_features = torch.tensor(query_feature).unsqueeze(0)
                support_features = torch.tensor(support_feature).unsqueeze(0)
                support_imgs = support_image_ten.unsqueeze(0)
                support_masks = support_target_ten.unsqueeze(0)
                grids = grid.unsqueeze(0)

            else:
                rand_value = torch.rand(1).item()
                if rand_value < self.args.pq:
                    support_image, support_target = query_image, query_target
                    support_image_name = query_image_name
                elif self.args.pq <= rand_value < self.args.pq + self.args.pr:
                    random_idx = torch.randint(len(self.train_ds), (1,)).item()
                    support_image, support_target = self.train_ds[random_idx]
                    support_image_name = self.train_ds.images[random_idx].split('/')[-1][:-4]
                else:
                    support_image_name = self.images_top50[query_image_name][sim_idx]
                    support_image, support_target = self.train_ds[self.train_ds.nameToNum[support_image_name]]
                
                support_label = support_target['labels'].numpy()[0]
                boxes = support_target['boxes'][torch.where(support_target['labels'] == support_label)[0]]
                support_image_copy = get_annotated_image(np.array(support_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
                support_image_copy_pil = Image.fromarray(support_image_copy)

                boxes = query_target['boxes'][torch.where(query_target['labels'] == label)[0]]
                query_image_copy = get_annotated_image(np.array(query_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
                query_image_copy_pil = Image.fromarray(query_image_copy)

                query_image_ten = self.transforms(query_image, None)[0]
                query_target_ten = self.transforms(query_image_copy_pil, None)[0]
                support_target_ten = self.transforms(support_image_copy_pil, None)[0]
                support_image_ten = self.transforms(support_image, None)[0]

                background_image = Image.new('RGB', (224, 224), color='white')
                background_image = self.background_transforms(background_image)
                grid = create_grid_from_images(background_image, support_image_ten, support_target_ten, query_image_ten,
                                                query_target_ten)
                support_img = support_image_ten.unsqueeze(0)
                support_mask = support_target_ten.unsqueeze(0)
                grid = grid.unsqueeze(0)
                _,  support_feature = self.load_feature(query_image_name,support_image_name)
                support_feature = torch.tensor(support_feature).unsqueeze(0)
                support_features = torch.cat((support_features,support_feature))
                support_imgs = torch.cat((support_imgs,support_img))
                support_masks = torch.cat((support_masks,support_mask))
                grids = torch.cat((grids,grid))

        batch = {'query_img': query_image_ten,
            'query_mask': query_target_ten,
            'support_imgs': support_imgs,
            'support_masks': support_masks,
            'grids': grids,
            'query_img_features': query_img_features,
            'support_features': support_features
        }
        return batch

