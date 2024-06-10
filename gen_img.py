"""
generate image in validation dataset, for calculating FID and IS
"""

# set args
model_ckpt = r"/root/data_new/zejia/workspace/psl/var/VAR/local_output_big/ar-ckpt-last.pth"
generated_folder_path = r'big_ep40'

data_path = r"/root/data_new/zejia/workspace/psl/var/datasets/COCO2017"
real_resize_folder_path = r'/root/data_new/zejia/workspace/psl/var/datasets/COCO2017/val_resize'
val_anno_path = r'/root/data_new/zejia/workspace/psl/var/datasets/COCO2017/annotations/val_images_anno.pkl'
cfg = 3.5
seed = 1103
more_smooth = False # True for more smooth output
tf32 = True
batchsize = 64

import os
import clip
import dist
import pickle
from PIL import Image
import os.path as osp
import torch, torchvision
from torchvision import transforms
import random
import numpy as np
from tqdm import tqdm
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from torch.utils.data import DataLoader
from models import VQVAE, build_vae_var
from utils.data import COCOdata, pil_loader, COCOCaptionFeature, ISImageDataset

MODEL_DEPTH = 20    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}

model_config = os.path.dirname(model_ckpt)
model_config = [o for o in os.listdir(model_config) if \
    os.path.isdir(os.path.join(model_config, o))]
assert len(model_config) == 1
model_config = model_config[0]
generated_folder_path = generated_folder_path + '_' + model_config
generated_folder_path = os.path.join(data_path, generated_folder_path)

print("[Info] Generated images will be saved to:", generated_folder_path)

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# download checkpoint
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpt = 'vae_ch160v4096z32.pth'
if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
if not osp.exists(model_ckpt): 
    raise ValueError('Path does not exist!')

# build vae, var
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums, depth=MODEL_DEPTH, shared_aln=False,
    )
    
# load checkpoints
var_ckpt_loaded = torch.load(model_ckpt, map_location='cpu')
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(var_ckpt_loaded['trainer']['var_wo_ddp'], strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)

epoch = var_ckpt_loaded['epoch']
iters = var_ckpt_loaded['iter']
print(f'[Info] ckpt_ep: {epoch}, ckpt_iter: {iters}')

print(f'[Info] prepare finished.')

# prepare validation data
val_text_feature = COCOCaptionFeature(data_path, 'val')
val_text_feature_loader = DataLoader(val_text_feature, batch_size=batchsize, 
                                        shuffle=False, num_workers=8, pin_memory=True)

# run faster
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

print('[Info] Loading validation data index...')
with open(val_anno_path, 'rb') as file:
    val_anno_data = pickle.load(file)
img_name_list = [val_anno_data[i][0].split('/')[-1] for i in range(len(val_anno_data))]
caption_list = [val_anno_data[i][1] for i in range(len(val_anno_data))]

if not os.path.exists(real_resize_folder_path):
    os.makedirs(real_resize_folder_path)
        
    print('[Info] Resizing real images...')
    
    for i in tqdm(range(len(val_text_feature))):
        real_img_path = val_anno_data[i][0]
        real_save_path = os.path.join(real_resize_folder_path, real_img_path.split('/')[-1])
        if os.path.exists(real_save_path):
            continue
        real_image = Image.open(real_img_path)
        real_resize_img = real_image.resize((299,299), PImage.LANCZOS)
        real_resize_img.save(real_save_path)

if not os.path.exists(generated_folder_path):
    os.makedirs(generated_folder_path)

for i, batch in enumerate(tqdm(val_text_feature_loader)):
    text_features_BD = batch.to(device)
    img_names = img_name_list[i * batchsize : (i+1) * batchsize]
    captions = caption_list[i * batchsize : (i+1) * batchsize]
    B = text_features_BD.shape[0]
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
            recon_B3HW = var.autoregressive_infer_cfg(B=B, text_features_BD=text_features_BD, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)  
        recons = recon_B3HW.permute(0, 2, 3, 1).mul_(255).cpu().numpy()
            
    for j in range(B):
        img_num = img_names[j]
        caption = captions[j]
        if (i * batchsize + j) % 500 == 0 or (i * batchsize + j) == 0:
            print(f'[Info] Generating image {i * batchsize + j}... Image name: {img_num}, Caption: {caption}')
        chw = PImage.fromarray(recons[j].astype(np.uint8)).resize((299,299), PImage.LANCZOS)
        generated_save_path = os.path.join(generated_folder_path, img_num)
        chw.save(generated_save_path)

print(f'[Info] Generarted images have been saved to {generated_folder_path}.')
print(f'[Info] Real images have been saved to {real_resize_folder_path}.')


# calculate IS and FID
print('[Info] Calculating IS...')
from utils.eval import inception_score
IS_mean, IS_std = inception_score(generated_folder_path, batch_size=batchsize, device=device)

print('[Result] IS is %.4f +- %.4f.' % (IS_mean, IS_std))

# clear everything in cuda memory
del vae, var
torch.cuda.empty_cache()

print('[Info] Calculating FID...')
os.system(f'python -m pytorch_fid {real_resize_folder_path} {generated_folder_path}')

