from models import *
import torch
import torchvision
import torchvision.transforms as transforms
from dataset import Dataset
from torch.optim import Adam
import models
from loss import DiceBCELoss
import pandas as pd
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import random

def output_visualization(inputs,path):

    print(inputs.size())
    array = inputs.detach().cpu().numpy()
    print(array.shape)

    for i in range(inputs.shape[0]):
        inputs[i] = inputs[i] * ((i+1)*50)

    mask = np.sum(array, axis = 0)
    mask = np.squeeze(mask)
    print(mask.shape)
    mask = mask[:,:,0]
    plt.figure()
    plt.imshow(mask)
    plt.show()
    mask = mask[:,:,1]
    plt.figure()
    plt.imshow(mask)
    plt.show()
    mask = mask[:,:,2]
    plt.figure()
    plt.imshow(mask)
    plt.show()
    mask = mask[:,:,3]
    plt.figure()
    plt.imshow(mask)
    plt.show()
    plt.savefig(path)



root = r"/homes/n20ravel/Documents/Data/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

out_channels = 4

model = UNet3D(in_channels=1, out_channels=out_channels)
model_path =  r"/homes/n20ravel/Documents/Data/epoch8_loss2.4024351239204407_model.pth"
model.load_state_dict(torch.load(model_path))

model.to(device)


criterion = DiceBCELoss()

transform = transforms.Compose(
    [transforms.RandomCrop(size=32)])

testset = pd.read_csv(os.path.join(root, "train.csv"))
testloader = Dataset(testset[1:2], train = False)

results_path = r"/homes/n20ravel/Documents/Data/Test_results"

transform = transforms.ToPILImage()


for i, data in enumerate(testloader, 1):
    inputs = data
    print(inputs.size())
    list_depth = list([*range(0,inputs.shape[2],1)])
    random.shuffle(list_depth)
    list_depth = list_depth[:16]
    print(list_depth)
    #inputs  = inputs[:,:,list_depth,:,:]
    #outputs = outputs[:,list_depth,:,:,:]
    #inputs = inputs[:,:,list_depth,:,:]
    inputs = inputs.to(device)
    #from torchsummary import summary

    #summary(model, (1,30,100,100))
    torch.cuda.empty_cache()
    predicted_outputs = model(inputs)
    predicted_outputs = torch.squeeze((predicted_outputs))
    predicted_outputs = predicted_outputs.permute(0,2,3,1)
    print(predicted_outputs.size())
    path = os.path.join(root,f"image{i}.png")
    output_visualization(predicted_outputs, path)    predicted_outputs = model(inputs)
