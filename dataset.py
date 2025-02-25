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
            print(e)
            exit()

    def  __len__(self):
        return self.N*self.N
        
    def __getitem__(self, i):
        ii,jj=divmod(i, self.N)
        scan_point=torch.Tensor((-self.width/2+self.step*ii,-self.width/2+self.step*jj,self.z)).flatten().to(self.device)
        hist=self.data[ii,jj,:]
        return {"hist":hist,"point":scan_point}

class NonconfDataset(Dataset):
    def __init__(self, data_path,z=0.0,device="cuda"):
        try:
            data_dict=loadmat(data_path)
            self.bin_resolution=data_dict["bin_resolution"]
            if type(self.bin_resolution)!=float:
                self.bin_resolution=self.bin_resolution[0][0]
            if self.bin_resolution<1e-9:
                self.bin_resolution=self.bin_resolution*3e8
            
            self.data=data_dict["data"] # [N,M]
            self.N=self.data.shape[0]
            self.M=self.data.shape[-1]
            self.device=device
            self.z=z

            self.data=torch.from_numpy(self.data).to(self.device)
            
            # 数据归一化
            self.data=self.data/torch.max(self.data)

            # 激光打在墙上的点
            self.laserPos=torch.from_numpy(data_dict["laserPos"]).to(self.device) # [N,3]

            # 激光出射位置
            if "laserOrigin" in data_dict:
                self.laserOrigin=torch.from_numpy(data_dict["laserOrigin"]).view(1,3).to(self.device)
            else:
                self.laserOrigin=None
            print(self.laserOrigin)
            # 相机对准在墙上的点
            self.cameraPos=torch.from_numpy(data_dict["cameraPos"]).view(1,3).to(self.device) # [1,3]

            # 相机位置
            if "cameraOrigin" in data_dict:
                self.cameraOrigin=torch.from_numpy(data_dict["cameraOrigin"]).view(1,3).to(self.device)
            else:
                self.cameraOrigin=self.cameraPos

            # 索引0对应的时间
            if data_dict["t0"]:
                self.t0=torch.from_numpy(data_dict["t0"]).item() # 浮点数
                print(self.t0)
            else:
                self.t0=0.0

        except  Exception as e:
            print(e)
            exit()

    def  __len__(self):
        return self.N
        
    def __getitem__(self, i):
        scan_point=self.laserPos[i,:]
        hist=self.data[i,:]
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
            self.grid=torch.from_numpy(data_dict["grid"]).to(self.device) # [N,3]
            print(self.grid.shape)

        except  Exception as e:
            print("no such file!")
            exit()

    def  __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, i):
        scan_point=self.grid[i,:]
        hist=self.data[i,:]
        return {"hist":hist,"point":scan_point}