import os
from matplotlib.image import imread
from torch.utils.data import Dataset
import numpy as np
import torch
from scipy.io import loadmat
from glob import glob
import re
import cv2
import random

import warnings

warnings.filterwarnings('error')

class ConfocalDataset(Dataset):
    def __init__(self, data_path,z=-2.0,device="cuda"):
        try:
            data_dict=loadmat(data_path)
            self.bin_resolution=data_dict["bin_resolution"]
            if type(self.bin_resolution)!=float:
                self.bin_resolution=self.bin_resolution[0][0]
            self.width=data_dict["width"]
            if type(self.width)!=float:
                self.width=self.width[0][0]
            
            self.data=data_dict["data"] # [N,N,M]
            self.N=self.data.shape[0]
            self.M=self.data.shape[-1]
            self.step=self.width/(self.N-1)
            self.device=device
            self.z=z

            # 这两个需要推出来
            self.obj_center=(0,0,0)
            self.obj_radius=1.1

            # 数据归一化
            # self.data=self.data/np.max(self.data)

            self.data=torch.from_numpy(self.data).to(self.device)

        except  Exception as e:
            print("no such file!")
            exit()

    def  __len__(self):
        return self.N*self.N
        
    def __getitem__(self, i):
        ii,jj=divmod(i, self.N)
        scan_point=(self.width/2-self.step*ii,self.width/2-self.step*jj,self.z)
        hist=self.data[ii,jj,:].reshape(-1)
        return {"hist":hist/torch.max(hist),"point":scan_point}