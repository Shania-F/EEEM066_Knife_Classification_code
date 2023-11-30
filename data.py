from torchvision.transforms.transforms import ColorJitter, RandomRotation, RandomVerticalFlip
from utils import *
from config import *
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as F
import pathlib
from torchvision.io import read_image
import numpy as np 
import cv2

import pandas as pd
import matplotlib.pyplot as plt
import random

# create dataset class
class knifeDataset(Dataset):
    def __init__(self,images_df,mode="train"):
        self.images_df = images_df.copy()
        self.images_df.Id = self.images_df.Id
        self.mode = mode

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self,index):
        X,fname = self.read_images(index)
        if not self.mode == "test":
            labels = self.images_df.iloc[index].Label
        else:
            y = str(self.images_df.iloc[index].Id.absolute())
        if self.mode == "train":
            X = T.Compose([T.ToPILImage(),
                    T.Resize((config.img_weight,config.img_height)),
                    # T.TrivialAugmentWide(),
                    T.ColorJitter(brightness=0.2,contrast=0,saturation=0,hue=0),
                    T.RandomRotation(degrees=(0, 180)),
                    T.RandomVerticalFlip(p=0.5),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(X)
        elif self.mode == "val":
            X = T.Compose([T.ToPILImage(),
                    T.Resize((config.img_weight,config.img_height)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(X)
        return X.float(),labels, fname

    def read_images(self,index):
        row = self.images_df.iloc[index]
        filename = str(row.Id)
        im = cv2.imread(filename)[:,:,::-1]  # OpenCV reads images in BGR format by default, so we change to RGB
        return im, filename


if __name__ == '__main__':
    train_imlist = pd.read_csv("train.csv")
    d1 = knifeDataset(train_imlist)

    num_images = 9
    random_indices = random.sample(range(5000), num_images)
    # Create a grid of subplots to display the images
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    fig.tight_layout()

    # Iterate through the dataset to get images and display them
    for i, index in enumerate(random_indices):
        image, label, filename = d1[index]

        # Transform the image tensor to a NumPy array and adjust dimensions
        image_np = T.ToPILImage()(image).convert("RGB")  # Convert tensor to PIL image
        axes[i // 3, i % 3].imshow(image_np)
        axes[i // 3, i % 3].set_title(f"Label: {label}")
        axes[i // 3, i % 3].axis('off')

    plt.savefig('VIS_data.png')
