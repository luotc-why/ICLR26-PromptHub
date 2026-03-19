import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import h5py
import torchvision.transforms.functional
import random

def convert_to_rgb(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

class DatasetColorization(Dataset):
    def __init__(self, datapath, image_transform, mask_transform, args = None,  padding: bool = 1,
                 random: bool = False, split: str = 'val', feature_name: str = 'features_vit-laion2b_pixel-level', seed: int = 0, simidx: int = 2, to_device: str = 'cuda:0'):
        self.padding = padding
        self.to_device = to_device
        self.args = args
        self.datapath = datapath
        self.random = random
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.support_img_path = os.path.join(datapath, 'train_data')
        self.support_mask_path = os.path.join(datapath, 'train_label')
        if split == 'val':
            self.query_img_path = os.path.join(datapath, 'test_data')
            self.query_mask_path = os.path.join(datapath, 'test_label')
            self.ds = self.build_img_metadata(f'{datapath}/test_data')
        else :
            self.query_img_path = os.path.join(datapath, 'train_data')
            self.query_mask_path = os.path.join(datapath, 'train_label')
            self.ds = self.build_img_metadata(f'{datapath}/train_data')
        self.split = split
        self.seed = seed
        np.random.seed(seed)
        self.feature_name = feature_name
        self.image_top50 = self.get_top50_images()
        self.simidx = simidx
        if split == 'val':
            self.img_feature_for_train_path = f'{datapath}/features_vit-laion2b_pixel-level_val/folder_query_features_by_vqgan_encoder.h5df'
        else :
            self.img_feature_for_train_path = f'{datapath}/features_vit-laion2b_pixel-level_trn/folder_query_features_by_vqgan_encoder.h5df'
        self.support_feature_for_train_path = f'{datapath}/features_vit-laion2b_pixel-level_trn/folder_support_features_by_vqgan_encoder.h5df'
    def __len__(self):
        return 50000

    def load_feature(self,query_name, support_name):
        with h5py.File(self.img_feature_for_train_path, "r") as f:
            query_img_feature = f[query_name][...]
        with h5py.File(self.support_feature_for_train_path, "r") as f:
            support_feature = f[support_name][...]
        return query_img_feature,support_feature

    def get_top50_images(self):
        if self.split == 'val':
            with open(f'{self.datapath}/features_vit-laion2b_pixel-level_val/new_top_50-similarity.json') as f:
                images_top50 = json.load(f)
        else :
            with open(f'{self.datapath}/features_vit-laion2b_pixel-level_trn/new_top_50-similarity.json') as f:
                images_top50 = json.load(f)
        return images_top50

    def create_grid_from_images(self, support_img, support_mask, query_img, query_mask):
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding,
                             2 * support_img.shape[2] + 2 * self.padding))
        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
        canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
        canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
        canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
        return canvas

    def build_img_metadata(self, img_dir):
        img_metadata = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        # print('Total %s images are: %d' % (img_dir, len(img_metadata)))
        return img_metadata

    def read_support_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.support_img_path, img_name))

    def read_query_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.query_img_path, img_name))

    def read_support_mask(self,img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.support_mask_path, img_name))
    
    def read_query_mask(self,img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.query_mask_path, img_name))

    def __getitem__(self, idx):
        query = self.ds[idx]
        grids = torch.tensor([]) 
        support_imgs = torch.tensor([]) 
        support_masks = torch.tensor([]) 
        query_img_features = torch.tensor([]) 
        support_features = torch.tensor([]) 
        query_image_ten, query_target_ten = self.mask_transform(self.read_query_img(query)), self.image_transform(self.read_query_mask(query))
        if self.split == 'val':
            query_image_name = query[:-5]
        else :
            query_image_name = query[:-4]
 
        for simidx in range(self.simidx):
            if simidx == 0:
                rand_val = torch.rand(1).item()
                if self.split == 'val':
                    support_name = self.image_top50[query[:-5]][simidx]
                    support = support_name+'.jpg'
                    support_image_ten, support_target_ten = self.mask_transform(self.read_support_img(support)), self.image_transform(self.read_support_mask(support))
                    grid = self.create_grid_from_images(support_image_ten, support_target_ten, query_image_ten, query_target_ten)
                    query_feature, support_feature = self.load_feature(query_image_name,support_name)
                else :
                    if rand_val < self.args.pq:
                        support_image_ten, support_target_ten = query_image_ten.clone(), query_target_ten.clone()
                        query_feature, support_feature = self.load_feature(query_image_name, query_image_name)
                        grid = self.create_grid_from_images(support_image_ten, support_target_ten, query_image_ten, query_target_ten)
                    elif self.args.pq <= rand_val < self.args.pq + self.args.pr:
                        random_idx = random.randint(0, len(self.ds) - 1)
                        support = self.ds[random_idx]
                        support_image_ten, support_target_ten = self.mask_transform(self.read_support_img(support)), self.image_transform(self.read_support_mask(support))
                        query_feature, support_feature = self.load_feature(query_image_name, support[:-4])
                        grid = self.create_grid_from_images(support_image_ten, support_target_ten, query_image_ten, query_target_ten)
                    else:
                        support_name = self.image_top50[query[:-4]][simidx]
                        support = support_name+'.jpg'
                        support_image_ten, support_target_ten = self.mask_transform(self.read_support_img(support)), self.image_transform(self.read_support_mask(support))
                        grid = self.create_grid_from_images(support_image_ten, support_target_ten, query_image_ten, query_target_ten)
                        query_feature, support_feature = self.load_feature(query_image_name,support_name)
                query_img_features = torch.tensor(query_feature).unsqueeze(0)
                support_features = torch.tensor(support_feature).unsqueeze(0)
                support_imgs = support_image_ten.unsqueeze(0)
                support_masks = support_target_ten.unsqueeze(0)
                grids = grid.unsqueeze(0)

            else:
                rand_val = torch.rand(1).item()
                if self.split == 'val':
                    support_name = self.image_top50[query[:-5]][simidx]
                    support = support_name+'.jpg'
                    support_image_ten, support_target_ten = self.mask_transform(self.read_support_img(support)), self.image_transform(self.read_support_mask(support))
                    grid = self.create_grid_from_images(support_image_ten, support_target_ten, query_image_ten, query_target_ten)
                    _, support_feature = self.load_feature(query_image_name,support_name)
                else :
                    if rand_val < self.args.pq:
                        support_image_ten, support_target_ten = query_image_ten.clone(), query_target_ten.clone()
                        _, support_feature = self.load_feature(query_image_name, query_image_name)
                        grid = self.create_grid_from_images(support_image_ten, support_target_ten, query_image_ten, query_target_ten)
                    elif self.args.pq <= rand_val < self.args.pq + self.args.pr:
                        random_idx = random.randint(0, len(self.ds) - 1)
                        support = self.ds[random_idx]
                        support_image_ten, support_target_ten = self.mask_transform(self.read_support_img(support)), self.image_transform(self.read_support_mask(support))
                        _, support_feature = self.load_feature(query_image_name, support[:-4])
                        grid = self.create_grid_from_images(support_image_ten, support_target_ten, query_image_ten, query_target_ten)
                    else:
                        support_name = self.image_top50[query[:-4]][simidx]
                        support = support_name+'.jpg'
                        support_image_ten, support_target_ten = self.mask_transform(self.read_support_img(support)), self.image_transform(self.read_support_mask(support))
                        grid = self.create_grid_from_images(support_image_ten, support_target_ten, query_image_ten, query_target_ten)
                        _, support_feature = self.load_feature(query_image_name,support_name)
                support_img = support_image_ten.unsqueeze(0)
                support_mask = support_target_ten.unsqueeze(0)
                support_feature = torch.tensor(support_feature).unsqueeze(0)
                grid = grid.unsqueeze(0)
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
