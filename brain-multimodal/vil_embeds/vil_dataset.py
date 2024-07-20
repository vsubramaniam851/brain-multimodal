import sys
import os
import torch
import numpy as np
import cv2
from PIL import Image

from torch.utils import data 
from torchvision import transforms
sys.path.append('vil_embeds/SLIP')
import utils

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
general_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

class ViLDataset(data.Dataset):
    def __init__(self, image_paths, contexts, use_cv2 = True, image_transforms = None, vis_processor = None, text_processor = None):
        self.image_paths = image_paths
        self.contexts = contexts
        self.use_cv2 = use_cv2
        self.image_transforms = image_transforms
        if not use_cv2:
            if image_transforms:
                if type(image_transforms) == list:
                    image_transforms = transforms.Compose(image_transforms)
                self.image_transforms = image_transforms
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.use_cv2:
            img = np.array(Image.open(self.image_paths[idx]))
            img = img[:, :, ::-1].copy()
        else:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            if self.image_transforms:
                if self.image_transforms != 'No transforms':
                    img = self.image_transforms(img)
            else:
                if self.vis_processor is None:
                    img = general_transform(img)
                else:
                    img = self.vis_processor['eval'](img)
        context = self.contexts[idx]
        if self.text_processor:
            context = self.text_processor['eval'](context)
        return {'image': img, 'context': context}