import os
import os.path as osp
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
from torch.utils.data import Dataset


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


class COCOdata(Dataset):
    def __init__(self, img_dir, anno_dir, mode="train", transform=None, loader=None, data_root=None):
        # self.img_dir_train = img_dir_train
        # self.img_dir_val = img_dir_val
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.transform = transform
        # self.classes = sorted(os.listdir(root_dir))
        # self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = self._load_images_anno(self.img_dir, mode)
        self.loader = loader
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize_01_into_pm1,
            ])
            
        if data_root is not None and osp.exists(osp.join(data_root, f'annotation_text_features_{mode}.npy')):
            self.text_features = np.load(osp.join(data_root, f'annotation_text_features_{mode}.npy'))
            assert self.text_features.shape[0] == len(self.images)
            self.load_text_features = True
        else:
            self.text_features = None
            self.load_text_features = False

    def _load_images_anno(self, img_dir, mode):
        if osp.exists(osp.join(self.anno_dir, f'{mode}_images_anno.pkl')):
            with open(osp.join(self.anno_dir, f'{mode}_images_anno.pkl'), 'rb') as f:
                images_anno = pickle.load(f)
            return images_anno
        else:
            images_anno = []
            
            if mode == 'train':
                anno_path = os.path.join(self.anno_dir, 'captions_train2017.json')
            else:
                anno_path = os.path.join(self.anno_dir,'captions_val2017.json')
            
            with open(anno_path, 'r') as f:
                    anno = json.load(f)
            
            for filename in tqdm(os.listdir(img_dir)):
                img_path = os.path.join(img_dir, filename)
                img_id = int(filename.split('.')[0])
                for i in anno['annotations']:
                    if i['image_id'] == img_id:
                        images_anno.append((img_path, i['caption']))
                        
            with open(osp.join(self.anno_dir, f'{mode}_images_anno.pkl'), 'wb') as f:
                pickle.dump(images_anno, f)
            
            return images_anno

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, caption = self.images[idx]
        image = self.loader(img_path)
        if self.transform:
            image = self.transform(image)
        if self.text_features is not None:
            return image, caption, torch.from_numpy(self.text_features[idx])
        else:
            return image, caption


def build_dataset(
    data_path: str, final_reso: int,
    hflip=False, mid_reso=1.125,
):
    # build augmentations
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_aug, val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ], [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
    img_dir_train = osp.join(data_path, 'train2017')
    img_dir_val = osp.join(data_path, 'val2017')
    anno_dir = os.path.join(data_path, 'annotations')
    # build dataset
    train_set = COCOdata(img_dir=img_dir_train, anno_dir=anno_dir, mode='train', loader=pil_loader, transform=train_aug, data_root=data_path)
    val_set = COCOdata(img_dir=img_dir_val, anno_dir=anno_dir, mode='val', loader=pil_loader, transform=val_aug, data_root=data_path)

    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}')
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')
    
    return train_set, val_set


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')