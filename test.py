# Author: Daiwei (David) Lu
# Make predictions on test images

import torch
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms
from model import Net
from utils import TestFile, Rescale, ToTensor, Normalize, show_dot
import argparse

MODEL_PATH = './model.pth'


def test_model(path):
    image = io.imread(path)
    image = transform.resize(image, (256, 256))

    device = torch.device("cuda")

    model = Net()
    model = model.to(device)

    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    with torch.no_grad():
        set = TestFile(path,
                       transform=transforms.Compose([
                           Rescale(256),
                           ToTensor(),
                           Normalize()
                       ])
                       )
        loader = torch.utils.data.DataLoader(set)
        for i, input in enumerate(loader):
            image = input['image'].float().cuda().to(device)
            coordinates = model(image).data
            coordinates = coordinates.cpu().numpy()
            print('{:.4f} {:.4f}'.format(coordinates[0][0], coordinates[0][1]))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Test Prediction')
    parser.add_argument('path', metavar='P', type=str,
                        help='path of file for prediction')
    args = parser.parse_args()
    test_model(args.path)


if __name__ == '__main__':
    main()
