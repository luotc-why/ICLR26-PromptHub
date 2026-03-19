"""Based on https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer/blob/main/data/pascal.py
"""
import os
from PIL import Image
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import sys
import random
current_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if current_path not in sys.path:
    sys.path.append(current_path)

from evaluate.mae_utils import PURPLE, YELLOW
import h5py

def create_grid_from_images_old(canvas, support_img, support_mask, query_img, query_mask):
    canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
    canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
    canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
    canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
    return canvas
mapped_dict = {
    '0': {'05': '05', '02': '02', '16': '15', '09': '09', '44': '40'},
    '1': {'06': '06', '03': '03', '17': '16', '62': '57', '21': '20'},
    '2': {'67': '61', '18': '17', '19': '18', '04': '04', '01': '01'},
    '3': {'64': '59', '20': '19', '63': '58', '07': '07', '72': '63'}
}


class DatasetMSCOCO(Dataset):
    def __init__(self, datapath, args, fold, split, image_transform, mask_transform, padding: bool = 1,
                 use_original_imgsize: bool = False, flipped_order: bool = False, have_tokens: bool = False,
                 reverse_support_and_query: bool = False, random: bool = False, ensemble: bool = False,
                 purple: bool = False, cluster: bool = False, feature_name: str = 'features_vit-laion2b_trn',
                 percentage: str = '', seed: int = 0, mode: str = '', arr: str = '', cls_base: bool = False,
                 selected_label: int = 0, simidx: int = 1):
        self.mscoco_pat = datapath
        self.fold = fold
        self.args = args
        self.split = split
        self.nfolds = 4
        self.flipped_order = flipped_order
        self.nclass = 20  # 20
        self.ncluster = 200
        self.padding = padding
        self.random = random
        self.ensemble = ensemble
        self.purple = purple
        self.cluster = cluster
        self.use_original_imgsize = use_original_imgsize
        self.cls_base = cls_base
        self.selected_label = selected_label

        self.img_path = os.path.join(datapath, 'trn2014')
        self.ann_path = os.path.join(datapath, 'Coco_Trainlabel')

        self.image_transform = image_transform
        self.reverse_support_and_query = reverse_support_and_query
        self.mask_transform = mask_transform
        self.data_path = datapath
        self.have_tokens = have_tokens
        # self.class_ids = self.build_class_ids()
        self.img_metadata_val = self.build_img_metadata('val') if '_val' in feature_name else self.build_img_metadata(
            'trn')
        self.img_metadata_trn = self.build_img_metadata('trn')
        self.all_img_metadata_trn = self.build_all_img_metadata('trn')
        # self.img_few_shot_metadata_trn = self.build_few_shot_metadata('trn', n_shot=16)

        self.feature_name = feature_name
        self.seed = seed
        self.percentage = percentage
        # self.images_top50_val = self.get_top50_images_val()
        self.images_top50_trn = self.get_top50_images_trn()
        self.images_top50_for_training = self.get_top50_images_for_training()
        self.img_feature_for_train_path = os.path.join(datapath, f'{self.feature_name}/folder{self.fold}_query_features_by_vqgan_encoder.h5df')
        self.support_feature_for_train_path = os.path.join(datapath, f'{self.feature_name}/folder{self.fold}_features_by_vqgan_encoder.h5df')
 
        self.mode = mode
        self.arr = arr
        self.simidx = simidx

    def __len__(self):
        return len(self.img_metadata_trn) if self.split == 'trn' else 1000

    def get_top50_images_for_training(self):
        with open(f"{self.mscoco_pat}/{self.feature_name}/folder{self.fold}_top_50-similarity.json") as f:
            images_top50 = json.load(f)

        images_top50_new = {}
        for img_name, img_class in self.img_metadata_trn:
            if img_name not in images_top50_new:
                images_top50_new[img_name] = {}
            images_top50_new[img_name]['top50'] = images_top50[str(img_name)]
            images_top50_new[img_name]['class'] = img_class

        return images_top50_new

    def get_top50_images_trn(self):
        images_top50_new = {}
        for img_name, img_class in self.img_metadata_trn:
            if img_name not in images_top50_new:
                images_top50_new[img_name] = {'class': []}

            # Check if img_class is not already in the list to avoid duplicates.
            if img_class not in images_top50_new[img_name]['class']:
                images_top50_new[img_name]['class'].append(img_class)

        return images_top50_new

    def create_gradiant_grid_images(self, support_img, support_mask, query_img, query_mask, arr):
        # create grid image for suppot images and query image.
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding,
                             2 * support_img.shape[2] + 2 * self.padding))

        content_list = [support_img, support_mask, query_img, query_mask]

        if arr == 'a1':
            support_img = content_list[0]
            support_mask = content_list[1]
            query_img = content_list[2]
            query_mask = content_list[3]

        elif arr == 'a2':
            support_img = content_list[1]
            support_mask = content_list[0]
            query_img = content_list[3]
            query_mask = content_list[2]

        elif arr == 'a3':
            support_img = content_list[3]
            support_mask = content_list[2]
            query_img = content_list[1]
            query_mask = content_list[0]

        elif arr == 'a4':
            support_img = content_list[2]
            support_mask = content_list[3]
            query_img = content_list[0]
            query_mask = content_list[1]

        elif arr == 'a5':
            support_img = content_list[1]
            support_mask = content_list[3]
            query_img = content_list[0]
            query_mask = content_list[2]

        elif arr == 'a6':
            support_img = content_list[3]
            support_mask = content_list[1]
            query_img = content_list[2]
            query_mask = content_list[0]

        elif arr == 'a7':
            support_img = content_list[2]
            support_mask = content_list[0]
            query_img = content_list[3]
            query_mask = content_list[1]

        elif arr == 'a8':
            support_img = content_list[0]
            support_mask = content_list[2]
            query_img = content_list[1]
            query_mask = content_list[3]

        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
        canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
        canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
        canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas

    def create_ensemble_grid_from_images(self, support_img, support_mask, query_img, query_mask, positions):
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding,
                             2 * support_img.shape[2] + 2 * self.padding))

        if positions == 'a1':
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
        elif positions == 'a2':
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_img
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_mask
        elif positions == 'a3':
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = support_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = query_mask
        elif positions == 'a4':
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_img
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = support_mask
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_mask
        elif positions == 'a5':
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_img
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_mask
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_mask
        elif positions == 'a6':
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = support_img
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = query_mask
        elif positions == 'a7':
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_img
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = support_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_mask
        elif positions == 'a8':
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas

    def create_all_grids(self, support_img, support_mask, query_img, query_mask):
        canvas_list = []
        # List of all possible arrangements
        arrangements = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']

        for arr in arrangements:
            canvas = self.create_ensemble_grid_from_images(support_img, support_mask, query_img, query_mask, arr)
            canvas_list.append(canvas)

        return canvas_list

    def get_tokens(self,query_name,support_name):
        with h5py.File(f'{self.mscoco_pat}/features_vit-laion2b_pixel-level_trn/folder_{self.fold}_gt_tokens.h5','r') as f:
            group = f.require_group(query_name)
            if support_name not in group:
                print(123,query_name,support_name)
                print('very bad! ')
                assert False
            return torch.tensor(f[query_name][support_name][:],dtype=torch.long)

    def __getitem__(self, idx):
        grids = torch.tensor([]) 
        support_imgs = torch.tensor([]) 
        support_masks = torch.tensor([]) 
        query_img_features = torch.tensor([]) 
        support_features = torch.tensor([]) 
        support_image_names = []
        vq_tokens = []
        for sim_idx in range(self.simidx):
            if sim_idx == 0:
                query_name, support_name, class_sample_query, class_sample_support = self.sample_episode_for_training(idx, sim_idx=sim_idx)
                rand_value = torch.rand(1).item()
                if rand_value < self.args.pq:
                    support_name = query_name
                    class_sample_support = class_sample_query
                elif self.args.pq <= rand_value < self.args.pq + self.args.pr:
                    random_idx = torch.randint(len(self.img_metadata_trn), (1,)).item()
                    support_name, _, class_sample_support, _ = self.sample_episode_for_training(random_idx, sim_idx=sim_idx)

                support_image_names.append(support_name)
                if self.have_tokens:
                    vq_tokens.append(self.get_tokens(query_name=query_name,support_name=support_name))
                query_img, query_cmask, support_img, support_cmask, org_qry_imsize = self.load_frame(query_name,
                                                                                                    support_name)
                name = query_name
                if self.image_transform:
                    query_img = self.image_transform(query_img)
                    query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask, class_sample_query,
                                                                        purple=self.purple)
                if self.mask_transform:
                    query_mask = self.mask_transform(query_mask)

                if self.image_transform:
                    support_img = self.image_transform(support_img)
                support_mask, support_ignore_idx = self.extract_ignore_idx(support_cmask, class_sample_support,
                                                                        purple=self.purple)

                if self.mask_transform:
                    support_mask = self.mask_transform(support_mask)

                if self.arr != 'ensemble':
                    grid = self.create_gradiant_grid_images(support_img, support_mask, query_img, query_mask, self.arr)

                else:
                    grid = self.create_all_grids(support_img, support_mask, query_img, query_mask)
                query_img_features, support_features = self.load_feature(query_name,support_name)
                query_img_features = torch.tensor(query_img_features).unsqueeze(0)
                support_features = torch.tensor(support_features).unsqueeze(0)
                support_imgs = support_img.unsqueeze(0)
                support_masks = support_mask.unsqueeze(0)
                grids = grid.unsqueeze(0)

            else:
                _query_name, _support_name, _class_sample_query, _class_sample_support = self.sample_episode_for_training(idx, sim_idx=sim_idx)
                rand_value = torch.rand(1).item()
                if rand_value < self.args.pq:
                    _support_name = _query_name
                    _class_sample_support = _class_sample_query
                elif self.args.pq <= rand_value < self.args.pq + self.args.pr:
                    random_idx = torch.randint(len(self.img_metadata_trn), (1,)).item()
                    _support_name, _, _class_sample_support, _ = self.sample_episode_for_training(random_idx, sim_idx=sim_idx)

                support_image_names.append(_support_name)
                if self.have_tokens:
                    vq_tokens.append(self.get_tokens(query_name=query_name,support_name=support_name))

                _, _, support_img, support_cmask, org_qry_imsize = self.load_frame(_query_name,_support_name)
                if self.image_transform:
                    support_img = self.image_transform(support_img)
                support_mask, support_ignore_idx = self.extract_ignore_idx(support_cmask, _class_sample_support,purple=self.purple)
                if self.mask_transform:
                    support_mask = self.mask_transform(support_mask)
                if self.arr != 'ensemble':
                    grid = self.create_gradiant_grid_images(support_img, support_mask, query_img, query_mask, self.arr)
                support_img = support_img.unsqueeze(0)
                support_mask = support_mask.unsqueeze(0)
                grid = grid.unsqueeze(0)
                query_img_feature,  support_feature = self.load_feature(_query_name,_support_name)
                support_feature = torch.tensor(support_feature).unsqueeze(0)
                support_features = torch.cat((support_features,support_feature))
                support_imgs = torch.cat((support_imgs,support_img))
                support_masks = torch.cat((support_masks,support_mask))
                grids = torch.cat((grids,grid))

        if self.have_tokens:
            vq_tokens = torch.stack(vq_tokens,dim = 0)
        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'grids': grids,
                 'query_image_name': name,
                 'support_image_names': support_image_names,
                 'vq_tokens': vq_tokens,
                 'query_img_features': query_img_features,
                 'support_features': support_features,
                 'vq_tokens': vq_tokens
                 }

        return batch


    def extract_ignore_idx(self, mask, class_id, purple):
        mask = np.array(mask)
        boundary = np.floor(mask / 255.)
        if not purple:
            mask[mask != class_id + 1] = 0
            mask[mask == class_id + 1] = 255
            return Image.fromarray(mask), boundary
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x, y] != class_id + 1:
                    color_mask[x, y] = np.array(PURPLE)
                else:
                    color_mask[x, y] = np.array(YELLOW)
        return Image.fromarray(color_mask), boundary

    def load_frame(self, query_name, support_name):
        # import pdb;pdb.set_trace()
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_img = self.read_img(support_name)
        support_mask = self.read_mask(support_name)
        org_qry_imsize = query_img.size

        return query_img, query_mask, support_img, support_mask, org_qry_imsize
    def load_feature(self,query_name, support_name):
        with h5py.File(self.img_feature_for_train_path, "r") as f:
            query_img_feature = f[str(query_name)+'.jpg'][...]
        with h5py.File(self.support_feature_for_train_path, "r") as f:
            support_feature = f[str(support_name)+'.jpg'][...]
        return query_img_feature,support_feature
    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = Image.open(os.path.join(self.ann_path, img_name) + '.png')
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode_for_training(self, idx, sim_idx):
        """Returns the index of the query, support and class."""
        query_name, class_sample = self.img_metadata_trn[idx]

        if self.cls_base:
            support_name = self.images_top50_for_training[query_name]['top50'][sim_idx]
            support_class = self.images_top50_trn[support_name]['class']
            while support_class != class_sample:
                sim_idx += 1
                if sim_idx >= len(self.images_top50_for_training[query_name]['top50']):
                    break
                support_name = self.images_top50_for_training[query_name]['top50'][sim_idx]
                support_class = self.images_top50_trn[support_name]['class']
        else:
            # print(query_name)
            support_name = self.images_top50_for_training[query_name]['top50'][sim_idx]
            # print(support_name)
            support_classes = self.images_top50_trn[str(support_name)]['class']
            if class_sample in support_classes:
                support_class = class_sample
            else:
                support_class = support_classes[0]

        if support_name == query_name:
            print('support_name = query_name ' + support_name)
            return self.sample_episode_for_training(idx, sim_idx + 1)

        if sim_idx >= len(self.images_top50_for_training[query_name]['top50']):
            print('query name: ', query_name)
            sim_idx = 0
            return self.sample_episode_for_training(idx + 1, sim_idx)

        return query_name, support_name, class_sample, support_class

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        return class_ids_val

    def build_few_shot_metadata(self, split, n_shot):

        def read_metadata(split, fold_id):
            cwd = self.args.save_base_dir

            fold_n_metadata_path = os.path.join(cwd, 'splits/pascal/%s/fold%d.txt' % (split, fold_id))

            with open(fold_n_metadata_path, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            # import pdb;pdb.set_trace()
            fold_n_metadata = [[data.split('__')[0], int(mapped_dict[str(self.fold)][(data.split('__')[1])]) -1 ] for data in fold_n_metadata]

            classes = {}
            for item in fold_n_metadata:
                if item[1] not in classes:
                    classes[item[1]] = []
                classes[item[1]].append(item)

            n = n_shot
            random.seed(1)
            selected_items = []
            for class_items in classes.values():
                selected_items.extend(random.sample(class_items, min(n, len(class_items))))

            # print(selected_items)

            return selected_items

        img_metadata = []
        img_metadata = read_metadata(split, 0)

        print('Total (%s) images are : %d' % (split, len(img_metadata)))

        return img_metadata

    def build_img_metadata(self, split):

        def read_metadata(split, fold_id):
            # cwd = os.path.dirname(os.path.abspath(__file__))
            cwd = self.args.save_base_dir

            if self.cluster:
                fold_n_metadata_path = os.path.join(cwd, 'split/coco/%s/fold_cluster%d.txt' % (split, fold_id))
            else:
                fold_n_metadata_path = os.path.join(cwd, 'split/coco/%s/fold%d.txt' % (split, fold_id))

            with open(fold_n_metadata_path, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]

            if self.cls_base:
                new_fold_n_metadata = []
                for data in fold_n_metadata:
                    label = int(data.split('__')[1]) - 1
                    if label + 1 == self.selected_label:
                        element = [data.split('__')[0], label]
                        new_fold_n_metadata.append(element)
            else:            
                new_fold_n_metadata = [[data.split('__')[0][:-4], int(mapped_dict[str(self.fold)][(data.split('__')[1])]) -1 ] for data in fold_n_metadata]

            return new_fold_n_metadata

        img_metadata = []
        img_metadata = read_metadata(split, self.fold)
        # print('Total (%s) images are : %d' % (split, len(img_metadata)))

        return img_metadata

    def build_all_img_metadata(self, split):

        def read_metadata(split, fold_id):
            # cwd = os.path.dirname(os.path.abspath(__file__))
            cwd = self.args.save_base_dir

            if self.cluster:
                fold_n_metadata_path = os.path.join(cwd, 'split/coco/%s/fold_cluster%d.txt' % (split, fold_id))
            else:
                fold_n_metadata_path = os.path.join(cwd, 'split/coco/%s/fold%d.txt' % (split, fold_id))

            with open(fold_n_metadata_path, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            # import pdb;pdb.set_trace()

            new_fold_n_metadata = [[data.split('__')[0], int(mapped_dict[str(self.fold)][(data.split('__')[1])]) -1 ] for data in fold_n_metadata]

            return new_fold_n_metadata

        img_metadata = []
        img_metadata = read_metadata(split, self.fold)

        return img_metadata
