import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2

class PNSDataset(Dataset):
    """PNSDataset"""
    def __init__(self, imgs, labels, transform=None):

        self.images = np.expand_dims(imgs, axis=1)
        self.labels = np.array(labels)
        self.transform = transform


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_tmp = self.images[idx][0]
        image = np.zeros((int(image_tmp.shape[0]), int(image_tmp.shape[1]), 3))
        for i in range(3):
            image[:,:,i] = image_tmp
        image = image.astype(np.uint8)
        res = self.transform(image=image)
        image = res['image']
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float()
        label = torch.tensor(int(self.labels[idx])).long()

        return image, label

class PNSClassDataset(Dataset):
    """PNSClassDataset"""
    def __init__(self, imgs, labels, img_class, transform=None):

        self.img_class = str(int(img_class))
        self.images = np.expand_dims(np.array(imgs), axis=1)
        self.labels = np.array(labels)
        self.transform = transform


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_tmp = self.images[idx][0]
        image = np.zeros((int(image_tmp.shape[0]), int(image_tmp.shape[1]), 3))
        for i in range(3):
            image[:,:,i] = image_tmp
        image = image.astype(np.uint8)
        res = self.transform(image=image)
        image = res['image']
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float()
        label = torch.tensor(int(self.labels[idx])).long()

        return image, label

class PNSTestDataset(Dataset):
    """PNSTestDataset"""
    def __init__(self, img, transform=None):

        self.images = np.expand_dims(np.array(img), axis=1)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_tmp = self.images[idx][0]
        image = np.zeros((int(image_tmp.shape[0]), int(image_tmp.shape[1]), 3))
        for i in range(3):
            image[:,:,i] = image_tmp
        image = image.astype(np.uint8)
        res = self.transform(image=image)
        image = res['image'].astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)

        return image





