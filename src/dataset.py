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
    if x.shape[-1]%2==0:
        x_padded = F.pad(x_mean, (pad_size, pad_size+1), mode='replicate')  # [N, M]
    else:
        x_padded = F.pad(x_mean, (pad_size, pad_size), mode='replicate')  # [N, M]
    
    # 如果输入长度为偶数，则长度会减小1
    return x_padded

def gaussian_filter(x, kernel_size=10, sigma=1.0):
    """
    在 dim=1 上对输入张量进行高斯滤波。

    参数:
        x (torch.Tensor): 输入张量，大小为 [N, M]。
        kernel_size (int): 高斯核大小。
        sigma (float): 高斯核的标准差。

    返回:
        torch.Tensor: 滤波后的张量，大小为 [N, M]。
    """

    # 创建 1D 高斯核
    kernel = torch.arange(kernel_size, dtype=torch.float32,device=x.device) - (kernel_size - 1) / 2.0
    kernel = torch.exp(-kernel**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # 扩展高斯核的维度以匹配输入张量
    kernel = kernel.view(1, 1, -1)  # [1, 1, kernel_size]

    # 使用 unfold 创建滑动窗口
    x_unfold = x.unfold(dimension=1, size=kernel_size, step=1)  # [N, M - kernel_size + 1, kernel_size]

    # 对每个窗口应用高斯核
    x_filtered = (x_unfold * kernel).sum(dim=2)  # [N, M - kernel_size + 1]

    # 处理边界：填充缺失的部分
    pad_size = (kernel_size - 1) // 2
    if x.shape[-1]!=pad_size*2+x_filtered.shape[-1]:
        x_padded = F.pad(x_filtered, (pad_size, pad_size+1), mode='replicate')  # [N, M]
    else:
        x_padded = F.pad(x_filtered, (pad_size, pad_size), mode='replicate')  # [N, M]

    return x_padded

class NLOSDataset(Dataset):
    def __init__(self, data_path,z=0.0,filter=False):
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
            self.z=z

            self.data=torch.from_numpy(self.data)
            self.data[:,:,-1]=0
            
            if filter:
                self.data=gaussian_filter(self.data.reshape(-1,self.M),kernel_size=10,sigma=1.0)
                self.data=self.data.view(self.N,self.N,-1)

            # 数据归一化
            self.data=self.data/torch.max(self.data)

            if "t0" in data_dict:
                self.t0=torch.from_numpy(data_dict["t0"]).item() # 浮点数
                print(self.t0)
            else:
                self.t0=0.0

        except  Exception as e:
            print(e)
            exit()

    def  __len__(self):
        return self.N*self.N
        
    def __getitem__(self, i):
        ii,jj=divmod(i, self.N)
        scan_point=torch.Tensor((-self.width/2+self.step*ii,-self.width/2+self.step*jj,self.z)).flatten()
        hist=self.data[ii,jj,:]
        return {"hist":hist,"point":scan_point}

class NonconfDataset(Dataset):
    def __init__(self, data_path,z=0.0):
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
            self.z=z

            self.data=torch.from_numpy(self.data)
            
            # maxvalue=torch.max(mean_filter(self.data,window_size=20))
            # print("histogram maxvalue:",maxvalue)
            # self.data=self.data/maxvalue
            self.data=self.data/torch.max(self.data)

            # 激光打在墙上的点
            self.laserPos=torch.from_numpy(data_dict["laserPos"]).float() # [N,3]

            # 激光出射位置
            if "laserOrigin" in data_dict:
                self.laserOrigin=torch.from_numpy(data_dict["laserOrigin"]).float().view(1,3)
            else:
                self.laserOrigin=None
            print(self.laserOrigin)
            # 相机对准在墙上的点
            self.cameraPos=torch.from_numpy(data_dict["cameraPos"]).float().view(1,3) # [1,3]

            # 相机位置
            if "cameraOrigin" in data_dict:
                self.cameraOrigin=torch.from_numpy(data_dict["cameraOrigin"]).float().view(1,3)
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

# 用于DDP的数据集
class PhfDataset(Dataset):
    def __init__(self, data_dir,z=0.0,filter=False):
        try:
            self.data_dir=data_dir

            data_path=data_dir+"setup.mat"
            data_dict=loadmat(data_path)
            self.bin_resolution=data_dict["bin_resolution"]
            if type(self.bin_resolution)!=float:
                self.bin_resolution=self.bin_resolution[0][0]
            if self.bin_resolution<1e-9:
                self.bin_resolution=self.bin_resolution*3e8
            self.filter=filter

            self.N=data_dict["N"]
            if type(self.N)!=float:
                self.N=self.N[0][0]
            self.M=data_dict["M"]
            if type(self.M)!=float:
                self.M=self.M[0][0]

            self.z=z

            data_path=data_dir+"indices2000.mat"
            if os.path.exists(data_path):
                indices=loadmat(data_path)["indices"]
                self.indices=[item[0] for item in indices]
            else:
                self.indices=[item for item in range(self.N)]

            # 激光打在墙上的点
            self.laserPos=torch.from_numpy(data_dict["laserPos"]).float() # [N,3]
            self.laserPos=self.laserPos[self.indices]

            # 激光出射位置
            if "laserOrigin" in data_dict:
                self.laserOrigin=torch.from_numpy(data_dict["laserOrigin"]).float().view(1,3)
            else:
                self.laserOrigin=None
            # print(self.laserOrigin)
            # 相机对准在墙上的点
            self.cameraPos=torch.from_numpy(data_dict["cameraPos"]).float().view(1,3) # [1,3]

            # 相机位置
            if "cameraOrigin" in data_dict:
                self.cameraOrigin=torch.from_numpy(data_dict["cameraOrigin"]).float().view(1,3)
            else:
                self.cameraOrigin=self.cameraPos

            # 索引0对应的时间
            if data_dict["t0"]:
                self.t0=torch.from_numpy(data_dict["t0"]).item() # 浮点数
                # print(self.t0)
            else:
                self.t0=0.0

        except  Exception as e:
            print(e)
            exit()

    def  __len__(self):
        return len(self.indices)
        
    def __getitem__(self, i):
        file=self.data_dir+f"{1+self.indices[i]}.mat"
        hist=loadmat(file)["img"]
        hist=torch.from_numpy(hist)
        if self.filter:
            hist=gaussian_filter(hist.reshape(-1,self.M),kernel_size=15,sigma=2.0)

        scan_point=self.laserPos[i,:]
        return {"hist":hist,"point":scan_point}

class RandomScanDataset(Dataset):
    def __init__(self, data_path):
        try:
            data_dict=loadmat(data_path)
            self.bin_resolution=data_dict["bin_resolution"]
            if type(self.bin_resolution)!=float:
                self.bin_resolution=self.bin_resolution[0][0]
            if self.bin_resolution<1e-9:
                self.bin_resolution=self.bin_resolution*3e8
            
            self.data=data_dict["data"] # [sample_num,M]
            self.N=64
            self.M=self.data.shape[-1]

            self.data=torch.from_numpy(self.data)
            self.data[:,-1]=0
            
            # 数据归一化
            self.data=self.data/torch.max(self.data)

            # 扫描网格
            self.grid=torch.from_numpy(data_dict["grid"]) # [N,3]

            # 索引0对应的时间
            if "t0" in data_dict.keys():
                self.t0=torch.from_numpy(data_dict["t0"]).item() # 浮点数
                print(self.t0)
            else:
                self.t0=0.0

        except  Exception as e:
            print(e)
            exit()

    def  __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, i):
        scan_point=self.grid[i,:]
        hist=self.data[i,:]
        return {"hist":hist,"point":scan_point}


class MultiViewDataset(Dataset):
    def __init__(self, data_path):
        try:
            data_dict=loadmat(data_path)
            self.bin_resolution=data_dict["bin_resolution"]
            if type(self.bin_resolution)!=float:
                self.bin_resolution=self.bin_resolution[0][0]
            if self.bin_resolution<1e-9:
                self.bin_resolution=self.bin_resolution*3e8
            
            self.data=data_dict["data"] # [sample_num,M]
            self.N=64

            self.data=torch.from_numpy(self.data)
            self.data[:,-1]=0
            
            # 数据归一化
            self.data=self.data/torch.max(self.data)

            # 扫描网格
            self.grid=torch.from_numpy(data_dict["grid"]) # [N,3]

            # 对应扫描面
            self.view_id=data_dict["view_id"] # [N,1], int

            # self.data=self.data[:128*128*2,151:350]
            # self.grid=self.grid[:128*128*2,:]
            # self.view_id=self.view_id[:128*128*2,:]

            # 索引0对应的时间
            if "t0" in data_dict.keys():
                self.t0=torch.from_numpy(data_dict["t0"]).item() # 浮点数
                print(self.t0)
            else:
                self.t0=0.0

            # self.t0=151*self.bin_resolution
            self.M=self.data.shape[-1]

        except  Exception as e:
            print(e)
            exit()

    def  __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, i):
        scan_point=self.grid[i,:]
        hist=self.data[i,:]
        return {"hist":hist,"point":scan_point,"view_id":self.view_id[i]}