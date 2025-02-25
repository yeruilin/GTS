import os
from matplotlib.image import imread
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import torch
from scipy.io import loadmat

import warnings

warnings.filterwarnings('error')

def mean_filter(x,window_size):
    # 数据先经过均值滤波再归一化
    x_unfold = x.unfold(dimension=1, size=window_size, step=1)  # [N, M - window_size + 1, window_size]

    # 计算每个窗口的均值
    x_mean = x_unfold.mean(dim=2)  # [N, M - window_size + 1]

    # 处理边界：填充缺失的部分
    # 因为滑动窗口会导致输出长度变小，所以需要在两端填充 (window_size - 1) / 2 个值
    pad_size = (window_size - 1) // 2
    x_padded = F.pad(x_mean, (pad_size, pad_size), mode='replicate')  # [N, M]
    
    # 如果输入长度为偶数，则长度会减小1
    return x_padded

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
            
            maxvalue=torch.max(mean_filter(self.data,window_size=20))
            print("histogram maxvalue:",maxvalue)
            self.data=self.data/maxvalue

            # 激光打在墙上的点
            self.laserPos=torch.from_numpy(data_dict["laserPos"]).float().to(self.device) # [N,3]

            # 激光出射位置
            if "laserOrigin" in data_dict:
                self.laserOrigin=torch.from_numpy(data_dict["laserOrigin"]).float().view(1,3).to(self.device)
            else:
                self.laserOrigin=None
            print(self.laserOrigin)
            # 相机对准在墙上的点
            self.cameraPos=torch.from_numpy(data_dict["cameraPos"]).float().view(1,3).to(self.device) # [1,3]

            # 相机位置
            if "cameraOrigin" in data_dict:
                self.cameraOrigin=torch.from_numpy(data_dict["cameraOrigin"]).float().view(1,3).to(self.device)
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