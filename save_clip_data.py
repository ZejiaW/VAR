import numpy as np
import os.path as osp
from tqdm import tqdm

import clip

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.data import build_dataset

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

DATA_ROOT = r"/root/data_new/zejia/workspace/psl/var/datasets/COCO2017"
train_set, val_set = build_dataset(data_path=DATA_ROOT, final_reso=224, hflip=False, mid_reso=1.125)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-L/14", device=device)

train_loader = DataLoader(train_set, batch_size=64, shuffle=False)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

all_text_val = None
iter = 0
with torch.no_grad():
    for _, text in tqdm(val_loader):
        text = clip.tokenize(text).to(device)
        text_features = model.encode_text(text).cpu()
        all_text_val = text_features if all_text_val is None else torch.cat([all_text_val, text_features], dim=0)
        del text, text_features
        iter += 1
        # if iter == 10:
        #     # generate a random number between 0 and len(all_text_val)
        #     i = np.random.randint(0, all_text_val.shape[0])
        #     sample = val_set[i][1]
        #     text = clip.tokenize(sample).to(device)
        #     text_features = model.encode_text(text).cpu()
            
        #     print(i, all_text_val.shape[0])
        #     print(all_text_val[i].shape, text_features.shape)
        #     assert torch.allclose(all_text_val[i], text_features[0], atol=1e-2)
        #     print(all_text_val[i][:10])
        #     print(text_features[0, :10])
        #     # exit(0)
            
all_text_val = all_text_val.numpy()
print("[INFO] Validation Text Features: ", all_text_val.shape)
np.save(osp.join(DATA_ROOT, "annotation_text_features_val.npy"), all_text_val)

del all_text_val

all_text_training = None
iter = 0
with torch.no_grad():
    for _, text in tqdm(train_loader):
        text = clip.tokenize(text).to(device)
        text_features = model.encode_text(text).cpu()
        all_text_training = text_features if all_text_training is None else torch.cat([all_text_training, text_features], dim=0)
        del text, text_features
        iter += 1
        # if iter == 10:
        #     # generate a random number between 0 and len(all_text_training)
        #     i = np.random.randint(0, all_text_training.shape[0])
        #     sample = train_set[i][1]
        #     text = clip.tokenize(sample).to(device)
        #     text_features = model.encode_text(text).cpu()
            
        #     print(i, all_text_training.shape[0])
        #     print(all_text_training[i].shape, text_features.shape)
        #     assert torch.allclose(all_text_training[i], text_features[0], atol=1e-2)
        #     print(all_text_training[i][:10])
        #     print(text_features[0, :10])
        #     # exit(0)
            
all_text_training = all_text_training.numpy()
print("[INFO] Training Text Features: ", all_text_training.shape)
np.save(osp.join(DATA_ROOT, "annotation_text_features_train.npy"), all_text_training)

