# Author: Daiwei (David) Lu
# Train custom model

from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from load import *
from model import ResNet, BasicBlock
from torchvision import models

import warnings

warnings.filterwarnings("ignore")
plt.ion()

def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=25):
    since = time.time()
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'VASC']
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.float().cuda().to(device), labels.long().cuda().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, torch.max(labels, 1)[1])

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.mean().backward()
                        optimizer.step()

                # statistics
                running_loss += loss.mean().item() * inputs.size(0)
                running_corrects += torch.sum(preds == torch.max(labels, 1)[1])
            # scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                scheduler.step(epoch_loss)
                val_loss.append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0:
            torch.save(best_model_wts, './trained_best.pth')
            plot(train_loss, val_loss, epoch)
            print('Saving...')
            print('Best val Acc: {:4f}'.format(best_acc))
        epoch_time = time.time() - epoch_start
        print('Epoch {} in {:.0f}m {:.0f}s'.format(epoch,
            epoch_time // 60, epoch_time % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss, val_loss

def plot(train, val, epoch):
    plt.plot(np.arange(len(train)), train, c='red', label='Training loss')
    plt.plot(np.arange(len(val)), val, c='blue', label='Validation loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./Loss Curve')
    plt.show()

image_datasets = {'train': TumorDataset(mode='train',
                                        transform=transforms.Compose([
                                            Rescale(256),
                                            RandomVerticalFlip(0.5),
                                            RandomHorizontalFlip(0.5),
                                            RandomColorJitter(0.9),
                                            ToTensor(),
                                            Normalize()
                                        ]),
                                        preload=True),
                  'val': TumorDataset(mode='test',
                                      transform=transforms.Compose([
                                          Rescale(256),
                                          ToTensor(),
                                          Normalize()
                                      ]),
                                      preload=True)}


def main():
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=128,
                                                        shuffle=True),
                   'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=128,
                                                      shuffle=True)}

    device = torch.device("cuda")

    #################################
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 64),
                                        nn.ReLU(),
                                        nn.Dropout(0.25),
                                        nn.Linear(64, 7))
    # model.fc = nn.Linear(num_ftrs, 7)
    model = model.to(device)
    #################################

    # model = ResNet(BasicBlock, [3,4,6,3])
    # model = model.to(device)
    class_count = np.sum(image_datasets['train'].labels, axis=0)
    
    beta = 0.99
    weights = beta ** class_count
    weights = (1 - beta) / (1 - weights)
    weight = torch.tensor(np.float32(weights)).cuda()
    
    criterion = nn.CrossEntropyLoss(weight, reduction='none')

    optimizer_conv = optim.Adam(model.parameters())
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_conv)
    print(model)
    epochs = 200
    model, train_loss, val_loss = train_model(model, criterion, optimizer_conv,
                                              exp_lr_scheduler, dataloaders, device, num_epochs=epochs)

    torch.save(model.state_dict(), './trained_best.pth')

    plot(train_loss, val_loss, epochs)

if __name__ == '__main__':
    main()
