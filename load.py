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
    def __init__(self, mode='test', transform=None):
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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root, self.images[idx] + '.jpg')
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'labels': self.labels[idx]}
        return image, torch.from_numpy(np.uint8(self.labels[idx]))

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
        self.mean = (0.5692824, 0.55365936, 0.5400631)
        self.std = (0.1325967, 0.1339596, 0.14305606)
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

# n=5
# dataset = pd.read_csv('labels/Train_labels.csv').values
# imgs = dataset[:,0]
# labels = dataset[:,1:]
# y = dataset[5]
# print('done')

# def imshow(image, title=None):
#     """Show image with landmarks"""
#     image = image.numpy().transpose((1, 2, 0))
#     inp = np.clip(image, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated
#
#
# dataset = TumorDataset('train',
#                         transform=transforms.Compose([
#                             Rescale(256),
#                             RandomVerticalFlip(.5),
#                             RandomHorizontalFlip(.5),
#                             RandomColorJitter(0.9),
#                             ToTensor()
#                         ]))
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
#                                              shuffle=True, num_workers=4)
# dataset_size = len(dataset)
# class_names = dataset.classes
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# # Get a batch of training data
# inputs, classes = next(iter(dataloader))
#
# import torchvision
#
# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs)
# imshow(out)
# imshow(out, title=[class_names[np.where(x==1)[0][0]] for x in classes])