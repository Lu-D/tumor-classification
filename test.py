# Author: Daiwei (David) Lu
# Make predictions on test images

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import argparse
from load import TumorImage, Rescale, ToTensor, Normalize
from model import ResNet, BasicBlock

MODEL_PATH = './resnet34pre771.pth'


def test_model(path):
    device = torch.device("cuda")
    model = ResNet(BasicBlock, [3,4,6,3])
    model = model.to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    with torch.no_grad():
        dataset = TumorImage(path,
                             transform=transforms.Compose([
                                 Rescale(256),
                                 ToTensor(),
                                 Normalize()
                             ]))

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        for input in dataloader:
            input = input.float().cuda().to(device)
            output = model(input)
            val = torch.max(output, 1)[1]
            return val.view(-1).cpu().numpy()[0]



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Test Prediction')
    parser.add_argument('path', metavar='P', type=str,
                        help='path of file for prediction')
    args = parser.parse_args()
    print(test_model(args.path))


if __name__ == '__main__':
    main()
