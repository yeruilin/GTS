import os
from matplotlib.image import imread
from torch.utils.data import Dataset
import numpy as np
import torch
from scipy.io import loadmat

import warnings

warnings.filterwarnings('error')

class ConfocalDataset(Dataset):
    def __init__(self, data_path,z=0.0,is_train=False,device="cuda"):
        try:
            data_dict=loadmat(data_path)
            self.bin_resolution=data_dict["bin_resolution"]
            if type(self.bin_resolution)!=float:
                self.bin_resolution=self.bin_resolution[0][0]
            if self.bin_resolution<1e-9:
                self.bin_resolution=self.bin_resolution*3e8
            self.width=data_dict["width"]
            if type(self.width)!=float:
                self.width=self.width[0][0]+0.0
            
            self.data=data_dict["data"] # [N,N,M]
            self.N=self.data.shape[0]
            self.M=self.data.shape[-1]
            self.step=self.width/(self.N-1)
            self.device=device
            self.z=z
            self.train=is_train

            self.data=torch.from_numpy(self.data).to(self.device)

            # 将深度衰减补上
            if self.train:
                grid_z=torch.linspace(0,self.M,self.M,dtype=torch.float32,device=self.device)*self.bin_resolution
                grid_z=grid_z.view(1,1,-1)
                self.data=self.data*(grid_z**2)
            
            # 数据归一化
            self.data=self.data/torch.max(self.data)

        except  Exception as e:
            print("no such file!")
            exit()

    def  __len__(self):
        return self.N*self.N
        
    def __getitem__(self, i):
        ii,jj=divmod(i, self.N)
        ii,jj=self.N//2,self.N//2
        scan_point=(-self.width/2+self.step*ii,-self.width/2+self.step*jj,self.z)
        hist=self.data[ii,jj,:].reshape(-1)
        return {"hist":hist,"point":scan_point}


class LCTDataset(Dataset):
    def __init__(self, data_path,z=0.0,subN=4,sample_mode=1,device="cuda"):
        try:
            data_dict=loadmat(data_path)
            self.bin_resolution=data_dict["bin_resolution"]
            if type(self.bin_resolution)!=float:
                self.bin_resolution=self.bin_resolution[0][0]
            if self.bin_resolution<1e-9:
                self.bin_resolution=self.bin_resolution*3e8
            self.width=data_dict["width"]
            if type(self.width)!=float:
                self.width=self.width[0][0]+0.0
            
            self.data=data_dict["data"] # [N,N,M]
            self.N=self.data.shape[0]
            self.M=self.data.shape[-1]
            self.step=self.width/(self.N-1)
            self.device=device
            self.sample_mode=sample_mode
            self.z=z
            self.subN=subN # 4*4作为子图
            self.subNum=self.N//self.subN # 子图数量

            self.data=torch.from_numpy(self.data).to(self.device)

        except  Exception as e:
            print("no such file!")
            exit()

    def  __len__(self):
        return self.subNum**2
        
    def __getitem__(self, k):
        if self.sample_mode==1: # 间隔采样
            start_i,start_j=divmod(k,self.subNum)

            scan_points=[]
            hist=torch.zeros((self.subN,self.subN,self.M),device=self.device,dtype=torch.float32)
            
            for i in range(self.subN):
                for j in range(self.subN):
                    ii=start_i+i*self.subNum
                    jj=start_j+j*self.subNum
                    scan_points.append((-self.width/2+self.step*ii,-self.width/2+self.step*jj,self.z))
                    hist[i,j,:]=self.data[ii,jj,:].reshape(-1)
            hist=hist.view(self.subN*self.subN,self.M)
        
        elif self.sample_mode==2: # 连续采样
            start_i,start_j=divmod(k,self.subNum)
            start_i=start_i*self.subN
            start_j=start_j*self.subN

            scan_points=[]
            hist=torch.zeros((self.subN,self.subN,self.M),device=self.device,dtype=torch.float32)
            
            for i in range(self.subN):
                for j in range(self.subN):
                    ii=start_i+i
                    jj=start_j+j
                    scan_points.append((-self.width/2+self.step*ii,-self.width/2+self.step*jj,self.z))
                    hist[i,j,:]=self.data[ii,jj,:].reshape(-1)
            hist=hist.view(self.subN*self.subN,self.M)

        return {"hist":hist/torch.max(hist),"point":scan_points}