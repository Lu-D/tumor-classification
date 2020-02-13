# Author: Daiwei (David) Lu
# A fully custom dataloader for the cellphone dataset

import os
import torch
import pandas as pd
from PIL import Image
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

import warnings

warnings.filterwarnings("ignore")
plt.ion()

class TumorDataset(Dataset):
    def __init__(self, mode='test', transform=None, preload=False):
        if mode == 'train':
            data = pd.read_csv('labels/Train_labels.csv')
            self.root = 'train'
        else:
            data = pd.read_csv('labels/Test_labels.csv')
            self.root = 'test'
        self.transform = transform
        self.classes = data.columns[1:]
        self.images = data.values[:,0]
        self.labels = data.values[:, 1:]
        self.preload = preload
        if preload:
            self.preloaded = []
            for i in range(len(self.images)):
                img_name = os.path.join(self.root, self.images[i] + '.jpg')
                self.preloaded.append(io.imread(img_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.preload:
            image = self.preloaded[idx]
        else:
            img_name = os.path.join(self.root, self.images[idx] + '.jpg')
            image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)
        # sample = {'image': image, 'labels': self.labels[idx]}
        return image, torch.from_numpy(np.uint8(self.labels[idx]))

class TumorImage(Dataset):
    def __init__(self, path, transform=None):
        img_name = os.path.join(path)
        self.image = [io.imread(img_name)]
        self.transform = transform

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.image[idx]
        if self.transform:
            image = self.transform(image)
        return image

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample
        new_h, new_w = self.output_size, self.output_size
        img = transform.resize(image, (new_h, new_w))
        return img


class Normalize(object):
    def __init__(self, inplace=False):
        #dataset mean/std
        # self.mean = (0.76964605, 0.54124683, 0.56347674)
        # self.std = (0.1364224, 0.15036866, 0.1672849)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.inplace = inplace

    def __call__(self, sample):
        return TF.normalize(sample, self.mean, self.std, self.inplace)


class ToTensor(object):

    def __call__(self, sample):
        dtype = torch.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        # image, labels = sample['image'], sample['labels']
        # return {'image': TF.to_tensor(image), 'labels': labels}
        return TF.to_tensor(sample)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image = sample
        if random.random() < self.p:
            image *= 255
            image = Image.fromarray(np.uint8(image))
            image = TF.hflip(image)
            image = np.array(image)
            image = np.double(image) / 255.
        return image


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image = sample
        if random.random() < self.p:
            image *= 255.
            image = Image.fromarray(np.uint8(image))
            image = TF.vflip(image)
            image = np.array(image)
            image = np.double(image) / 255.
        return image


class RandomColorJitter(object):
    def __init__(self, p=0.2, brightness=(0.5, 1.755), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.1, 0.1)):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample):
        image = sample
        if random.random() < self.p:
            image *= 255.
            image = Image.fromarray(np.uint8(image))
            modifications = []

            brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            modifications.append(transforms.Lambda(lambda img: TF.adjust_brightness(image, brightness_factor)))

            contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            modifications.append(transforms.Lambda(lambda img: TF.adjust_contrast(image, contrast_factor)))

            saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            modifications.append(transforms.Lambda(lambda img: TF.adjust_saturation(image, saturation_factor)))

            hue_factor = random.uniform(self.hue[0], self.hue[1])
            modifications.append(transforms.Lambda(lambda img: TF.adjust_hue(image, hue_factor)))

            random.shuffle(modifications)
            modification = transforms.Compose(modifications)
            image = modification(image)

            image = np.array(image)
            image = np.double(image) / 255.
        return image