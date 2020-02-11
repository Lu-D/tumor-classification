# Author: Daiwei (David) Lu
# Useful Utils

import torch
from skimage import io, transform
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import utils
import torchvision.transforms.functional as TF


class TestFile(Dataset):

    def __init__(self, file, transform=None):
        self.file = file
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        sample = io.imread(self.file)
        if self.transform:
            sample = self.transform(sample)
        return sample


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        new_h, new_w = self.output_size, self.output_size
        img = transform.resize(sample, (new_h, new_w))
        return img


class Normalize(object):
    def __init__(self, inplace=False):
        self.mean = (0.5692824, 0.55365936, 0.5400631)
        self.std = (0.1325967, 0.1339596, 0.14305606)
        self.inplace = inplace

    def __call__(self, sample):
        image = sample
        return {'image': TF.normalize(image, self.mean, self.std, self.inplace),
                'original': image}


class ToTensor(object):

    def __call__(self, sample):
        dtype = torch.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        image = sample.transpose((2, 0, 1))
        return torch.from_numpy(image).type(dtype)


def show_dot(image, coordinates):
    plt.imshow(image)
    plt.scatter(image.shape[1] * coordinates[0][0], image.shape[0] * coordinates[0][1], marker='.', c='r')
    plt.pause(0.001)


def batch_show(sample_batched):
    """Show image for a batch of samples."""
    images_batch, coordinates_batch = \
        sample_batched['original'], sample_batched['coordinates']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        # plt.scatter(coordinates_batch[i, 0].cpu().numpy() * 256 + i * im_size + (i + 1) * grid_border_size,
        #             coordinates_batch[i, 1].cpu().numpy() * 256 + grid_border_size,
        #             marker='.', c='r')
        plt.title('Batch from dataloader')


def visualize_model(model, dataloaders, device):
    was_training = model.training
    model.eval()
    fig = plt.figure()

    with torch.no_grad():
        for i, batch in enumerate(dataloaders['val']):
            inputs, labels = batch['image'], batch['coordinates']
            inputs = inputs.float().cuda().to(device)
            print('Label:', batch['coordinates'].data)
            batch['coordinates'].data = model(inputs).data
            plt.figure()
            batch_show(batch)
            plt.axis('off')
            plt.ioff()
            plt.show()

        model.train(mode=was_training)

def find_norm():
    dataset = TumorDataset('data/labels/train.txt',
                           'data',
                           mode='/train',
                           transform=transforms.Compose([
                               Rescale(256)
                           ]))
    loader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=1,
        shuffle=False
    )

    pixel_mean = np.zeros(3)
    pixel_std = np.zeros(3)
    k = 1
    for load in loader:
        imgs = load['image']
        imgs = np.array(imgs)
        print(imgs.shape)
        for i in range(imgs.shape[0]):
            image = imgs[i]
            pixels = image.reshape((-1, image.shape[2]))

            for pixel in pixels:
                diff = pixel - pixel_mean
                pixel_mean += diff / k
                pixel_std += diff * (pixel - pixel_mean)
                k += 1

    pixel_std = np.sqrt(pixel_std / (k - 2))
    print(pixel_mean)
    print(pixel_std)