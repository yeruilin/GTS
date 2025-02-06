import os
from matplotlib.image import imread
from torch.utils.data import Dataset
import numpy as np
import torch
from scipy.io import loadmat

import warnings

warnings.filterwarnings('error')

class NLOSDataset(Dataset):
    def __init__(self, data_path,z=0.0,device="cuda",confocal=True):
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

            self.data=torch.from_numpy(self.data).to(self.device)
            self.data[:,:,-1]=0
            
            # 数据归一化
            self.data=self.data/torch.max(self.data)

            if not confocal:
                self.laserPosition=[item for sublist in data_dict["laserPosition"] for item in sublist]
                print(self.laserPosition)

        except  Exception as e:
            print("no such file!")
            exit()

    def  __len__(self):
        return self.N*self.N
        
    def __getitem__(self, i):
        ii,jj=divmod(i, self.N)
        scan_point=(-self.width/2+self.step*ii,-self.width/2+self.step*jj,self.z)
        hist=self.data[ii,jj,:]
        return {"hist":hist,"point":scan_point}

# NLOS dataset with arbitary scanning pattern
class RandomScanDataset(Dataset):
    def __init__(self, data_path,device="cuda"):
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
            
            self.data=data_dict["data"] # [sample_num,M]
            self.N=64
            self.M=self.data.shape[-1]
            self.device=device

            self.data=torch.from_numpy(self.data).to(self.device)
            self.data[:,-1]=0
            
            # 数据归一化
            self.data=self.data/torch.max(self.data)

            # 扫描网格
            self.grid=data_dict["grid"]
            print(self.grid.shape)

        except  Exception as e:
            print("no such file!")
            exit()

    def  __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, i):
        scan_point=(self.grid[i,0],self.grid[i,1],self.grid[i,2])
        hist=self.data[i,:]
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