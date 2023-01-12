import torch
import torchvision
import torchvision.transforms as transforms
from dataset import Dataset
from torch.optim import Adam
import models 
from loss import DiceBCELoss
import pandas as pd
import os
import sys
import random

def saveModel(epoch, loss, savePath): 
    path = os.path.join(savePath, f"epoch{epoch}_loss{loss}_model.pth")
    torch.save(model.state_dict(), path) 


def train(model, criterion, optimizer, trainloader, testloader, N_epoch, savePath, device, transform):
    
    best_val_loss = 10**10
    
    for epoch in range(N_epoch):  # loop over the dataset multiple times

        print(f"epoch {epoch} is starting")
        
        running_loss = 0.0
        running_val_loss = 0.0 
        
        for i, data in enumerate(trainloader, 1):
            print("iterations",i)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            list_depth = list([*range(0,inputs.shape[2],1)])
            random.shuffle(list_depth)
            list_depth = list_depth[:16]
            print(list_depth)
            inputs  = inputs[:,:,list_depth,:,:]
            print("inputs", inputs.size())
            labels = labels[:,list_depth,:,:,:]
            inputs, labels = inputs.to(device), labels.to(device)


            # zerinputso the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            print("out: ", outputs.shape)
            print("lab :", labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statisticsrunning_val_loss
            running_loss += loss.item()
            inputs = inputs.detach().cpu()
            outputs = outputs.detach().cpu()
            labels = labels.detach().cpu()
            del inputs
            del labels
            torch.cuda.empty_cache()

            with torch.no_grad():
                print("eval")
                model.eval() 
                for data in testloader:
                    inputs, outputs = data
                    list_depth = list([*range(0,inputs.shape[2],1)])
                    random.shuffle(list_depth)
                    list_depth = list_depth[:16]
                    print(list_depth)
                    inputs  = inputs[:,:,list_depth,:,:]
                    outputs = outputs[:,list_depth,:,:,:]

                    inputs = inputs.to(device)
                    outputs = outputs.to(device)
                    predicted_outputs = model(inputs) 
                    val_loss = criterion(predicted_outputs, outputs)
                    print("val_loss")
                    print(val_loss)
                    running_val_loss += val_loss.item()
                    inputs = inputs.detach().cpu()
                    outputs = outputs.detach().cpu()
                    del inputs
                    del outputs
                    torch.cuda.empty_cache()

        # Save the model if the accuracy is the best 
        if running_val_loss <= best_val_loss:
            saveModel(epoch, running_val_loss, savePath) 
            best_val_loss = running_val_loss 
         
        # Print the statistics of the epoch 
        print('Completed training batch', epoch, 'Training Loss is: %.4f' %running_loss, 'Validation Loss is: %.4f' %running_val_loss)



if __name__ == "__main__":
    root = r"/homes/n20ravel/Documents/Data/"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    torch.cuda.empty_cache()
    savePath = r"/homes/n20ravel/Documents/Data/"
    model = models.UNet3D(in_channels=1, out_channels=4)
    model.to(device)
    criterion = DiceBCELoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    transform = transforms.Compose(
        [transforms.RandomCrop(size=32)])
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 1
    N_epoch = 10


    #Test set don't have ground masks, so we can't use it for validation
    # testset = pd.read_csv(os.path.join(root, "test.csv"))
    # testloader = Dataset(testset, transform=transform)

    trainset = pd.read_csv(os.path.join(root, "train.csv"))

    trainloader = Dataset(trainset[:2])

    testloader = Dataset(trainset[16:])


    
    train(model, criterion, optimizer, trainloader, testloader, N_epoch, savePath, device, transform)
    print('Finished Training')