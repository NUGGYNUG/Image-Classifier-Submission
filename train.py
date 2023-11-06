import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision.models as models
import json
from workspace_utils import active_session

import argparse

parser = argparse.ArgumentParser('change default parameters and settings')
parser.add_argument('--save_dir', type=str, help='save_directory to /opt (default: current)')
parser.add_argument('--arch', type=str, help='use vgg19 architecture (default: vgg19)')
parser.add_argument('--learning_rate', type=float, help='change learning rate (default: 0.001)')
parser.add_argument('--epochs', type=int, help='change number of epochs (default: 3)')
parser.add_argument('--gpu', type=str, help='use gpu (y/n) if available (default: n)')

args = parser.parse_args()

data_dir = 'flowers'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Create datasets for training, validation, and testing
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}

# Create data loaders for each dataset
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=0) for x in ['train', 'valid', 'test']}

# Get the size of each dataset
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
class_names = image_datasets['train'].classes

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    
#Build and train your network

if args.gpu is None:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.arch is None:
    model = models.vgg19(pretrained=True)
    model_name = 'vgg19'

for param in model.parameters():
    param.requires_grad = False
    
model = models.vgg19(weights=True)
model

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p = 0.5)),
                          ('fc2', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

for param in model.parameters():
    param.requires_grad = False
    
model.classifier = classifier

def train_model(model, criterion, optimizer, sched,   
                                      num_epochs=25, device="cuda"):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.to(device)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                sched.step()
                model.train() 
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

eps=4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device={}".format(device))
model.to(device)
                
# Save the checkpoint
if args.save_dir is None:
    torch.save({'classifier':classifier, 'model':model.state_dict(), 'lr': lr_value, 'epoch':epochs}, model_name)
else:
    path = os.path.join(args.save_dir, '{}_epochs{}.pth'.format(model_name, epochs))
    torch.save({'classifier':classifier, 'model':model.state_dict(), 'lr': lr_value, 'epoch':epochs}, path)

print ('Model saved.  Ready to predict.')
