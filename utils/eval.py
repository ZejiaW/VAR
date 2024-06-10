import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.stats import entropy
from PIL import Image
from tqdm import tqdm

from .data import ISImageDataset

def inception_score(image_path, batch_size=64, num_workers=4, splits=10, device='cuda'):
    
    assert batch_size > 0

    # Set up dataloader
    dataset = ISImageDataset(image_path, transforms_=[
        transforms.Resize((299, 299), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)
    
    N = len(dataset)
    assert N > batch_size
    print('[Info] Total number of generated images:', N)

    # Load the Inception v3 model
    inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).eval().to(device)
    # inception_model.fc = nn.Identity()  # Remove final classification layer

    def get_predictions(x):
        with torch.no_grad():
            x = inception_model(x)
        return torch.softmax(x, dim=1).detach().cpu().numpy()

    # Get predictions for the images
    print('[Info] Computing predictions using inception v3 model...')
    preds = np.zeros((N, 1000))
    for i, batch in enumerate(tqdm(dataloader)):
        batch = batch.type(torch.FloatTensor).to(device)
        preds[i * batch_size : i * batch_size + batch.size(0)] = get_predictions(batch)

    # Compute the inception score
    print('[Info] Computing KL Divergence...')
    split_scores = []
    for k in tqdm(range(splits)):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)