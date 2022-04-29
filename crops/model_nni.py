import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torch.functional as F
from torch.utils.data import DataLoader, Dataset
import nni

from dataset_model_nni import CropDataset, CropClassifier


def main(params):

    # read data and save it in a list: data = [[img, label]]
    
    # directories, where images are saved
    img_size = 224
    data = [] # contains [img, label]
    classes_ct = {}
    batch_size = 32
    epochs = 100

    path = 'data/crop_images'
    folders = ['jute', 'maize', 'rice', 'sugarcane', 'wheat']
    classes = {'jute': 0, 'maize': 1, 'rice': 2, 'sugarane': 3, 'wheat': 4}

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

    # define transformations (normalize and augment the data)
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

    X = np.array([i[0] for i in data])
    y = np.array([i[1] for i in data])

    # define train and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)

    print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
    print(f'X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}\n')

    # save as dataset
    train_dataset = CropDataset(X_train, y_train, transform=transforms_train)
    valid_dataset = CropDataset(X_valid, y_valid, transform=transforms_valid)

    # DataLoader to create train_loader and valid_loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    print('Batches in training data')
    for batch, (data, target) in enumerate(train_loader):
        print(f'Batch Nr {batch}, data shape {data.shape}, target.shape {target.shape})')

    print('\n')
    hidden_size = params['hidden_size'] 
    conv_size1, conv_size2, conv_size3 = params['conv_size1'], params['conv_size2'], params['conv_size3']
    model = CropClassifier(hidden_size=hidden_size, conv_size1=conv_size1, conv_size2=conv_size2, 
                            conv_size3=conv_size3,img_size=img_size)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # train and validate the model
    valid_loss_min = np.Inf
    correct_valid = [0 for i in range(len(folders))]
    total_valid = [0 for i in range(len(folders))]
    for epoch in range(epochs):

        train_loss = 0.0
        valid_loss = 0.0

        # train the model
        model.train()
        for data, target in train_loader:
            # clear all gradients of optimized variables
            optimizer.zero_grad()
            # forward pass
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass
            loss.backward()
            # optimization step
            optimizer.step()

            train_loss += loss

        train_loss /= len(train_loader.sampler)

        # validate the model
        model.eval()
        for data, target in valid_loader:
            # make predictions
            output = model(data)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)    
            # calculate loss
            loss = criterion(output, target)

            # calculate accuracy
            correct = pred.eq(target)
            for i in range(batch_size):
                length_batch = len(target)
                # last batch might be smaller
                if i >= length_batch:
                    break
                label = target[i].item()
                total_valid[label] += 1
                correct_valid[label] += correct[i].item()

            valid_loss += loss

        valid_loss /= len(valid_loader.sampler)

        if epoch % 10 == 0:
            print(f'epoch {epoch}/{epochs} - train loss: {train_loss} - valid loss: {valid_loss}')
            nni.report_intermediate_result(float(valid_loss)) 

        # save model if loss is decreasing
        if valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), 'model_crop.pt')
    print(f"Final Loss: {valid_loss}")
    nni.report_final_result(float(valid_loss))

    print('\n')
    for i in range(len(folders)):
        if total_valid[i] > 0:
            print(f'Validation accuracy of {folders[i]}: {correct_valid[i]/total_valid[i]*100:.2f}')
        else:
            print(f'Validation accuracy of {folders[i]}: N/A (no training examples)')



if __name__=='__main__':
    try:
        # get parameters form tuner
        params = nni.get_next_parameter()
        print(params)
        main(params)
    except Exception as exception:
        #logger.exception(exception)
        raise
