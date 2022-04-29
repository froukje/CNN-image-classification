from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch
import nni

class CropDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

class CropClassifier(nn.Module):
    def __init__(self, img_size, hidden_size, conv_size1, conv_size2, conv_size3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, conv_size1, 3) # input channels, output, kernel size
        self.conv2 = nn.Conv2d(conv_size1, conv_size2, 3)
        self.conv3 = nn.Conv2d(conv_size2, conv_size3, 3)

        # find out the size of the following fc layer
        x = torch.randn(3, img_size, img_size).view(-1, 3, img_size, img_size)
        self._to_linear = None
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 5) # 5 classes

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        if self._to_linear == None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2] # x[0] first element of batch

        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear) # flatten x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
