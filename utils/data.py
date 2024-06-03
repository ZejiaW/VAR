import os
import os.path as osp
import json

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
from torch.utils.data import Dataset


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


# def build_dataset(
#     data_path: str, final_reso: int,
#     hflip=False, mid_reso=1.125,
# ):
#     # build augmentations
#     mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
#     train_aug, val_aug = [
#         transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
#         transforms.RandomCrop((final_reso, final_reso)),
#         transforms.ToTensor(), normalize_01_into_pm1,
#     ], [
#         transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
#         transforms.CenterCrop((final_reso, final_reso)),
#         transforms.ToTensor(), normalize_01_into_pm1,
#     ]
#     if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
#     train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
#     # build dataset
#     train_set = DatasetFolder(root=osp.join(data_path, 'train'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=train_aug)
#     val_set = DatasetFolder(root=osp.join(data_path, 'val'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=val_aug)
#     num_classes = 1000
#     print(f'[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
#     print_aug(train_aug, '[train]')
#     print_aug(val_aug, '[val]')
    
#     return num_classes, train_set, val_set


class COCOdata(Dataset):
    def __init__(self, img_dir, anno_dir, mode, loader, transform=None):
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.transform = transform
        # self.classes = sorted(os.listdir(root_dir))
        # self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = self._load_images_anno(self.img_dir, mode)
        self.loader = loader

    def _load_images_anno(self, img_dir, mode):
        images_anno = []
        for filename in os.listdir(img_dir):
            img_path = os.path.join(img_dir, filename)
            img_id = int(filename.split('.')[0])
            if mode == 'train':
                anno_path = os.path.join(self.anno_dir, 'captions_train2017.json')
            else:
                anno_path = os.path.join(self.anno_dir,'captions_val2017.json')
            with open(anno_path, 'r') as f:
                anno = json.load(f)
            for i in anno['annotations']:
                if i['image_id'] == img_id:
                    images_anno.append((img_path, i['caption']))
        return images_anno

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, caption = self.images[idx]
        image = self.loader(img_path)
        if self.transform:
            image = self.transform(image)
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
    anno_dir = 'annotations'
    # build dataset
    train_set = COCOdata(img_dir=osp.join(data_path, 'train2017'), anno_dir=anno_dir,mode='train', loader=pil_loader, transform=train_aug)
    val_set = COCOdata(img_dir=osp.join(data_path, 'val2017'), anno_dir=anno_dir,mode='val', loader=pil_loader, transform=val_aug)

    # train_set = DatasetFolder(root=osp.join(data_path, 'train'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=train_aug)
    # val_set = DatasetFolder(root=osp.join(data_path, 'val'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=val_aug)
    # num_classes = 1000
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