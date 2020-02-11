# Author: Daiwei (David) Lu
# Train custom model

from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from load import *
from model import Net
from utils import visualize_model

import warnings

warnings.filterwarnings("ignore")
plt.ion()


# def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=25):
#     since = time.time()
#     dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
#     class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'VASC']
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_loss = 9001.
#     best_acc = 1.
#     train_loss = []
#     val_loss = []
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)
#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()  # Set model to evaluate mode
#             running_loss = 0.0
#             running_corrects = 0.0
#             # Iterate over data.
#             for loader in dataloaders[phase]:
#                 inputs, labels = loader['image'], loader['coordinates']
#                 inputs = inputs.float().cuda().to(device)
#                 labels = labels.float().cuda().to(device)
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#                 # forward
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#
#                 # backward + optimize only if in training phase
#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()
#
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += calc_acc(labels, outputs)
#             if phase == 'train':
#                 scheduler.step()
#
#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects / dataset_sizes[phase]
#
#             print('{} Loss: {:.6f}'.format(
#                 phase, epoch_loss))
#             print('{} Acc: {:.6f}'.format(
#                 phase, epoch_acc
#             ))
#             if phase == 'train':
#                 train_loss.append(epoch_loss)
#             else:
#                 val_loss.append(epoch_loss)
#
#             # deep copy the model
#             if phase == 'val' and epoch_acc < best_acc:
#                 best_loss = epoch_loss
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())
#
#         print()
#
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Loss: {:6f}'.format(best_loss))
#     print('Best val Acc: {:6f}'.format(best_acc))
#
#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model, train_loss, val_loss

def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=25):
    since = time.time()
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'VASC']
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs):
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
            count = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                count += 1
                print(count)
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
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == torch.max(labels, 1)[1])
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                val_loss.append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss, val_loss

image_datasets = {'train': TumorDataset(mode='train',
                                        transform=transforms.Compose([
                                            Rescale(256),
                                            # RandomVerticalFlip(0.5),
                                            # RandomHorizontalFlip(0.5),
                                            # RandomColorJitter(0.9),
                                            ToTensor()
                                        ])),
                  'val': TumorDataset(mode='test',
                                      transform=transforms.Compose([
                                          Rescale(256),
                                          # RandomVerticalFlip(0.1),
                                          # RandomHorizontalFlip(0.1),
                                          # RandomColorJitter(0.1),
                                          ToTensor()
                                      ]))}


def main():
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=128,
                                                        shuffle=True),
                   'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=128,
                                                      shuffle=True)}

    device = torch.device("cuda")

    model = Net()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_conv = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.5 every 20 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=20, gamma=0.5)

    print(model)
    epochs = 100
    model, train_loss, val_loss = train_model(model, criterion, optimizer_conv,
                                              exp_lr_scheduler, dataloaders, device, num_epochs=epochs)

    torch.save(model.state_dict(), './trainedmodel.pth')
    visualize_model(model, dataloaders, device)
    plt.ioff()
    plt.show()

    plt.plot(np.arange(epochs), train_loss, c='red', label='Training loss')
    plt.plot(np.arange(epochs), val_loss, c='blue', label='Validation loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./Loss Curve')

    # torch.save(model.state_dict(), './trainedmodel.pth')


if __name__ == '__main__':
    main()
