import torch
import torch.nn as nn
import torch.optim as optim

import os
import time
from random import randint
from tqdm import tqdm


import numpy as np
from scipy import stats
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold

import nibabel as nib
import pydicom as pdm
import nilearn as nl
import nilearn.plotting as nlplt
import h5py

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as anim
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import seaborn as sns
import imageio
from skimage.transform import resize
from skimage.util import montage

from IPython.display import Image as show_gif
from IPython.display import clear_output
from IPython.display import YouTubeVideo

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss


import albumentations as A
from albumentations import Compose, HorizontalFlip
# from albumentations.pytorch import ToTensorV2 

import warnings
warnings.simplefilter("ignore")



class BratsDataset(Dataset):
    def __init__(self,df: pd.DataFrame, phase: str="predict", is_resize: bool=False, filename=None):
        self.phase = phase
        self.augmentations = get_augmentations(phase)
        self.data_types = ['flair.nii', 't1.nii', 't1c.nii', 't2.nii']
        self.is_resize = is_resize
        self.df = df
        self.filename = filename
        
    # def __len__(self):
    #     return self.df.shape[0]
    
    # def get_data(self, data):
    #     self.data = data
    #     # load all modalities
    #     images = []
    #     for modality in self.data:
    #         print(modality)
    #         img_path = f'/content/{modality}'
    #         img = self.load_img(img_path)#.transpose(2, 0, 1)
            
    #         if self.is_resize:
    #             img = self.resize(img)
    
    #         img = self.normalize(img)
    #         images.append(img)
    #     img = np.stack(images)
    #     img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))
                
    #     return {
    #         "image": img,
    #     }
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        # id_ = self.df.loc[idx, 'Brats20ID']
        # root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]
        # load all modalities
        root_path = 'static/upload/'
        images = []
        for data_type in self.filename:
            img_path = os.path.join(root_path, data_type)
            img = self.load_img(img_path)#.transpose(2, 0, 1)
            # print(type(img))
            if self.is_resize:
                img = self.resize(img)
    
            img = self.normalize(img)
            images.append(img)
        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))
        id_ = 0
        print(type(img),len(img))
        return {
            "Id": id_,
            "image": img,
        }

    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data.astype(np.float64)
    
    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)
    
    def resize(self, data: np.ndarray):
        data = resize(data, (78, 120, 120), preserve_range=True)
        return data
    

def get_augmentations(phase):
    list_transforms = []
    
    list_trfms = Compose(list_transforms)
    return list_trfms


def get_dataloader(
    dataset: torch.utils.data.Dataset,
    phase: str,
    fold: int = 0,
    batch_size: int = 1,
    num_workers: int = 4,
    filename = None
):
    '''Returns: dataloader for the model training'''
    # df = pd.read_csv(path_to_csv)
    
    # train_df = df.loc[df['fold'] != fold].reset_index(drop=True)
    # val_df = df.loc[df['fold'] == fold].reset_index(drop=True)

    # df = train_df if phase == "train" else val_df

    data = [['0']]
 
    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns = ['Name'])

    dataset = dataset(df, phase, filename=filename)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,   
    )
    return dataloader


class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
          )

    def forward(self,x):
        return self.double_conv(x)

    
class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.encoder(x)

    
class Up(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    
class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)


class UNet3d(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask


# Specify a path to save to
# PATH = "/content/unet-v2.pth"

# # Save
# # torch.save(net.state_dict(), PATH)

# # Load
# device = torch.device('cpu')
# model = UNet3d(in_channels=4, n_classes=3, n_channels=24)
# model.load_state_dict(torch.load(PATH, map_location=device))


# model = torch.load('/content/unet-v2.pth')
# model = UNet3d(in_channels=4, n_classes=3, n_channels=24)
# model.load_state_dict(torch.load("/content/unet-v2.pth", map_location='cuda'))
# model=model.cuda()

class UNetV2:
    def __init__(self):
        self.model = UNet3d(in_channels=4, n_classes=3, n_channels=24)
        self.model.load_state_dict(torch.load("segmentation/saved_models/unet-v2.pth", map_location='cuda'))
        self.model=self.model.cuda()
        # self.model = torch.load('segmentation/saved_models/unet-v2-without-gpu.pth')
        # self.data_loader = DataLoad()
        # self.model = model
        
   

    def predict(self,filename,threshold = 0.33) :
      val_dataloader = get_dataloader(BratsDataset,filename = filename, phase='valid', fold=0)
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      print(device)
      # device = 'cpu'
      results = {"Id": [],"image": [], "Prediction": []}
      # self.threshold = threshold
      with torch.no_grad():
          for i, data in enumerate(val_dataloader):
              id_, imgs = data['Id'], data['image']
              imgs = imgs.to(device)
              print(type(imgs))
              logits = self.model(imgs.float())
              probs = torch.sigmoid(logits)
              print(type(probs))
              print(i)
              predictions = (probs >= threshold).float()
              print(type(predictions))
              predictions =  predictions.cpu()
              print(type(predictions))
              
              results["Id"].append(id_)
              results["image"].append(imgs.cpu())
              results["Prediction"].append(predictions)
              # only 5 pars
              if (i > 5):    
                  return results
          return results

