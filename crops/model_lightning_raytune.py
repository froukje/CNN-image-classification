import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune

from dataset_model import CropDataset

class CropDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config["batch_size"]

    def setup(self, stage):
        # transforms
        transforms_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])

        transforms_valid = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
            
        # prepare transformations

        data = [] # contains [img, label]
        classes_ct = {}
        for idx, folder in enumerate(folders):
            for file in os.listdir(os.path.join(path, folder)): # all images
                img_path = os.path.join(path, folder, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (img_size, img_size))
                # get labels (can be defined by folder name)
                label = idx
                data.append([img, idx])
                # count images in classes
                if label not in classes_ct.keys():
                    classes_ct[label] = 0
                else:
                    classes_ct[label] +=1

        print(f'Classes: {classes}')
        print(f'Nr of images in each class: {classes_ct}\n')

        X = np.array([i[0] for i in data])
        y = np.array([i[1] for i in data])

        # define train and validation set
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)

        print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
        print(f'X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}\n')

        # save as dataset
        self.train_dataset = CropDataset(X_train, y_train, transform=transforms_train)
        self.valid_dataset = CropDataset(X_valid, y_valid, transform=transforms_valid)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=8)

class CropClassifierLightning(pl.LightningModule):
    def __init__(self, config, img_size):
        super().__init__()
        self.conv_layer1 = config['conv_layer1']
        self.conv_layer2 = config['conv_layer2']
        self.conv_layer3 = config['conv_layer3']
        self.hidden_layer = config['hidden_layer']
        self.lr = config['lr']
        self.batch_size = config['batch_size']

        print(self.conv_layer1)

        self.conv1 = nn.Conv2d(3, self.conv_layer1, 3) # input channel, output, kernel size
        self.conv2 = nn.Conv2d(self.conv_layer1, self.conv_layer2, 3)
        self.conv3 = nn.Conv2d(self.conv_layer2, self.conv_layer3, 3)

        # find size of following fc layer
        x = torch.randn(3, img_size, img_size).view(-1, 3, img_size, img_size) # first dim is batch size
        self._to_linear = None
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear, self.hidden_layer)
        self.fc2 = nn.Linear(self.hidden_layer, 5) # 5 classes
        self.accuracy = pl.metrics.Accuracy()

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        if self._to_linear == None:
            self._to_linear = np.prod(x[0].shape)

        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear) # flatten x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.log_softmax(x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        acc = self.accuracy(logits, y)
        self.log('train_loss', loss)
        self.log('train_accuracy', acc)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch  
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        acc = self.accuracy(logits, y)
        self.log('valid_loss', loss)
        return {'val_loss': loss, 'val_accuracy': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer

def train_crop(config, data_dir=None, num_epochs=10, num_gpus=0):
    
    data_module = CropDataModule(config)
    model = CropClassifierLightning(config, img_size)
    metrics = {'loss': 'val_loss', 'acc': 'val_accuracy'}
    callbacks = [TuneReportCallback(metrics, on='validation_end')]
    trainer = pl.Trainer(max_epochs=100, callbacks=callbacks)
    trainer.fit(model, data_module)

    torch.save(model.state_dict(), 'model_crop_lightning.pt')

if __name__ == '__main__':

    # directories, where images are saved
    path = '/home/frauke/Crop_Images/data/crop_images'
    folders = ['jute', 'maize', 'rice', 'sugarcane', 'wheat']

    img_size = 224
    classes = {'jute': 0, 'maize': 1, 'rice': 2, 'sugarane': 3, 'wheat': 4}

    num_samples = 10
    num_epochs = 10
    gpus_per_trial = 0

    # Defining a search space
    config = {"conv_layer1": tune.choice([32, 64, 128]),
                "conv_layer2": tune.choice([64, 128, 256]),
                "conv_layer3": tune.choice([64, 128, 256]),
                "hidden_layer": tune.choice([128, 256, 512]),
                "lr": tune.loguniform(1e-4, 1e-1),
                "batch_size": tune.choice([32, 64, 128])
    }
    
    # Execute the hyperparameter search
    trainable = tune.with_parameters(train_crop, 
                                    num_epochs=num_epochs,
                                    num_gpus=gpus_per_trial)

    analysis = tune.run(trainable,
                        resources_per_trial={
                        "cpu": 8,
                        "gpu": gpus_per_trial
                        },
                        metric="loss",
                        mode="min",
                        config=config,
                        num_samples=num_samples,
                        name="tune_crop")


    #train_crop(config)
    print(analysis.best_config)


