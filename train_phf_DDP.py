### 测试分布式训练
# DDP在所有卡上的参数都是完全共享的，这里我们只是对像素进行了不同的网格采样，代入计算loss，使得拟合的更准确。
# 所以最后只需要保存一个卡上的参数就可以了

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import matplotlib.pyplot as plt
import os
import math
import scipy
import argparse
from dataset import *
from data_utils import plot_hist

import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

class GaussianModel(nn.Module):
    def __init__(self, num_points, scale,
                 bin_resolution=0.01,num_bins=512,t0=0,decay=2.0,confocal=True,
                 laserOrigin=None,cameraPos=[0,0,0],cameraOrigin=[0,0,0] # nonconfocal
                 ):
        super().__init__()

        self.confocal=confocal
        self.num_points=num_points

        # activation function
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        # Initial parameters
        self.colours=nn.Parameter(0.1*torch.ones((self.num_points,1), dtype=torch.float32))
        self.pre_act_opacities = nn.Parameter(8.0 * torch.ones((self.num_points,1), dtype=torch.float32))
        self.pre_act_scales = nn.Parameter(self.scaling_inverse_activation(torch.ones_like(self.pre_act_opacities)*scale))

        # Used by both confocal and nonconfocal
        self.bin_resolution=bin_resolution
        self.num_bins=num_bins
        self.t0=t0
        self.decay=decay

        # Used in nonconfocal
        self.laserOrigin=laserOrigin
        self.cameraPos=cameraPos
        self.cameraOrigin=cameraOrigin

    def parameters(self):
        return [self.pre_act_scales, self.colours, self.pre_act_opacities]
    
    def get_all_parameters(self):
        return {
            'scale': self.get_scaling.clone().detach(),
            'rho': self.get_colour.clone().detach(),
            'o': self.get_opacity.clone().detach()
        }
    
    def __len__(self):
        return self.num_points
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self.pre_act_scales)
    @property
    def get_opacity(self):
        return self.opacity_activation(self.pre_act_opacities)
    @property
    def get_colour(self):
        return self.colours**2
    
    # 利用极坐标的形式计算histogram
    def render_conf_hist2(self,means, scan_point):
        # 计算强度
        intensity=self.get_colour.flatten() # (N,)

        # 计算两组基的系数
        coeff=self.get_opacity.flatten().unsqueeze(1) # (N,1)

        # 计算片元中心的深度:scan_point is [1,3]
        r0 = torch.norm(means-scan_point, p=2, dim=1).unsqueeze(1) # (N,1)

        r_=self.t0/2+self.bin_resolution/2*torch.arange(1,1+self.num_bins,dtype=torch.float32).to(scan_point.device).flatten() # (M,)
        r=r_.view(1,self.num_bins) #(1,M)

        sigma=torch.mean(self.get_scaling,dim=1).unsqueeze(1) # (N,1)
        sigma=torch.clip(sigma,self.bin_resolution/2) #一定要不小于分辨率才能保证数值稳定

        # 概率密度,[N,M]
        pdf1=math.sqrt(0.5/math.pi)*torch.exp(-0.5*((r-r0)/sigma)**2)/sigma # 高斯分布
        pdf2=(r-r0)*torch.exp(-0.5*((r-r0)/sigma)**2)/(sigma**2) # 瑞利分布
        pdf=coeff*pdf1+(1-coeff)*pdf2 # 两个分布叠加
        pr=pdf*self.bin_resolution/2 # 概率, [N,M]
        pr=torch.clip(pr,0,1)

        # print(torch.mean(torch.sum(pr,1)))

        hist=intensity.unsqueeze(1)*pr # (N,M)
        hist=torch.sum(hist,dim=0).flatten()
        hist=hist/torch.pow(r_,self.decay)

        return hist

    def render_nonconf_hist2(self, means,laserPos):
        # 计算强度
        intensity=self.get_colour.flatten() # (N,)

        # 计算两组基的系数
        coeff=self.pre_act_opacities.flatten().unsqueeze(1) # (N,1)

        # 计算激光点和相机点到两边的距离
        r0_=torch.norm(self.cameraPos-self.cameraOrigin, p=2, dim=1)
        r0_-=self.t0
        if self.laserOrigin!=None:
            r0_+=torch.norm(laserPos-self.laserOrigin, p=2, dim=1)

        # 计算片元中心的深度
        a = torch.norm(means-laserPos, p=2, dim=1).unsqueeze(1) # (N,1)
        b = torch.norm(means-self.cameraPos, p=2, dim=1).unsqueeze(1) # (N,1)
        r0=a+b+r0_ # (N,1)

        r_=self.bin_resolution*torch.arange(1,1+self.num_bins,dtype=torch.float32).to(laserPos.device).flatten() # (M,)
        r=r_.view(1,self.num_bins) #(1,M)

        sigma=torch.mean(self.get_scaling,dim=1).unsqueeze(1) # (N,1)
        sigma=torch.clip(sigma,self.bin_resolution/2) #一定要不小于分辨率才能保证数值稳定

        # 概率密度,[N,M]
        pdf1=math.sqrt(1/math.pi)*torch.exp(-((r-r0)**2/4/sigma**2))/2/sigma # 高斯分布
        pdf2=(r-r0)*torch.exp(-((r-r0)**2/4/sigma**2))/(2*sigma**2) # 瑞利分布
        pdf=coeff*pdf1+(1-coeff)*pdf2 # 两个分布叠加
        pr=pdf*self.bin_resolution
        # print(torch.sum(pr,dim=1))
        pr=torch.clip(pr,0,1)

        hist=intensity.unsqueeze(1)/((torch.pow(a,self.decay)*(torch.pow(b,self.decay)))) # (N,1)
        hist=hist*pr # (N,M)
        hist=torch.sum(hist,dim=0).flatten()

        return hist

    def forward(self,means,scan_point):
        if self.confocal:
            return self.render_conf_hist2(means,scan_point)
        else:
            return self.render_nonconf_hist2(means,scan_point)

def gather_all_parameters(rank, world_size, local_params,pixels):
    """
    收集所有GPU上的参数到主GPU
    返回: 主GPU上包含所有参数的字典
    """
    # 为每个参数创建存储列表
    scale_list = [torch.zeros_like(local_params['scale']) for _ in range(world_size)]
    rho_list = [torch.zeros_like(local_params['rho']) for _ in range(world_size)]
    o_list = [torch.zeros_like(local_params['o']) for _ in range(world_size)]
    
    # 收集所有GPU的参数
    dist.all_gather(scale_list, local_params['scale'])
    dist.all_gather(rho_list, local_params['rho'])
    dist.all_gather(o_list, local_params['o'])
    
    if rank == 0:
        # 在主GPU上拼接所有参数
        rho = torch.zeros(pixels[0], pixels[1], pixels[2], dtype=torch.float32, device="cpu")
        rho[0::2, 0::2,:] = rho_list[0].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()# 按照奇偶位置插入
        rho[0::2, 1::2,:] = rho_list[1].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()
        rho[1::2, 0::2,:] = rho_list[2].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()
        rho[1::2, 1::2,:] = rho_list[3].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()

        scale = torch.zeros(pixels[0], pixels[1], pixels[2], dtype=torch.float32, device="cpu")
        scale[0::2, 0::2,:] = scale_list[0].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()
        scale[0::2, 1::2,:] = scale_list[1].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()
        scale[1::2, 0::2,:] = scale_list[2].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()
        scale[1::2, 1::2,:] = scale_list[3].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()

        o = torch.zeros(pixels[0], pixels[1], pixels[2], dtype=torch.float32, device="cpu")
        o[0::2, 0::2,:] = o_list[0].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()
        o[0::2, 1::2,:] = o_list[1].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()
        o[1::2, 0::2,:] = o_list[2].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()
        o[1::2, 1::2,:] = o_list[3].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()
        
        return {"rho":rho.numpy(),"scale":scale.numpy(),"o":o.numpy()}
    
    # if rank == 0:
    #     # 在主GPU上拼接所有参数
    #     rho = local_params['rho'].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu().numpy()
    #     scale = local_params['scale'].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu().numpy()
    #     o = local_params['o'].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu().numpy()
    #     return {"rho":rho,"scale":scale,"o":o}
    
    return None

def save_parameters(all_params, save_dir="temp"):
    os.makedirs(save_dir, exist_ok=True)

    numpy_params = {k: v.numpy() for k, v in all_params.items()}
    torch.save(numpy_params, os.path.join(save_dir, "result.npz"))
    
    print(f"All parameters saved to {save_dir}")


def makegrid(minimalpos, maximalpos, gridsize,rank,world_size):
    assert world_size==4
    xx=rank//2
    yy=rank-2*xx

    minimalpos = np.asarray(minimalpos)
    maximalpos = np.asarray(maximalpos)
    gridsize = np.asarray(gridsize)

    # Number of pixels per direction
    pixels = np.ceil(np.abs(minimalpos - maximalpos) /2/gridsize).astype(int)*2

    # Unit vectors scaled by grid size
    vx = np.array([1, 0, 0]) * gridsize
    vy = np.array([0, 1, 0]) * gridsize
    vz = np.array([0, 0, 1]) * gridsize

    # Generate grid points
    pts = []
    for x in range(xx,pixels[0],2):
        for y in range(yy,pixels[1],2):
            for z in range(pixels[2]):
                tmp = minimalpos + x * vx + y * vy + z * vz
                pts.append(tmp)

    pts = np.array(pts)
    return pts,pixels


def train(rank, args):
    # init DDP
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12345',
        rank=rank,
        world_size=args.world_size
    )

    dataset = PhfDataset(args.data_path,filter=False)
    bin_resolution=dataset.bin_resolution
    cameraOrigin=dataset.cameraOrigin.to(rank)
    cameraPos=dataset.cameraPos.to(rank)
    if dataset.laserOrigin!=None:
        laserOrigin=dataset.laserOrigin.to(rank)

    num_bins=dataset.M
    confocal=False
    decay=1
    scale=0.005
    
    # 场景参数
    min_pos=[-1.3,0.5,0.65] ## phasor_id5的参数
    max_pos=[0.0,1.8,0.95]
    grid_size=[0.0075,0.0075,0.0075]
    scale=0.002
    # min_pos=[-0.5,-0.5,0.85] ## phasor_id11的参数
    # max_pos=[0.8,0.8,1.1]
    # grid_size=[0.0075,0.0075,0.0075]
    # scale=0.002
    # decay=2
    # min_pos=[-1.8,0.5,0.4] ## office phasor id4的参数
    # max_pos=[0.0,1.8,1.3]
    # grid_size=[0.0075,0.0075,0.012]
    # scale=0.002
    # min_pos=[-1.2,-0.55,0.5] ## office phasor id1-3的参数
    # max_pos=[0.8,0.65,1.4]
    # grid_size=[0.0075,0.0075,0.012]
    # scale=0.002

    if rank==0:
        print("min_pos:",min_pos)
        print("max_pos:",max_pos)
        print("grid_size:",grid_size)
        print("bin resolution:",bin_resolution)
        print("cameraOrigin:",cameraOrigin)
        print("cameraPos:",cameraPos)
        print("t0:",dataset.t0)
    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler)
    train_itr = iter(train_loader)
    
    # 生成当前GPU上的高斯中心
    xyz,pixels=makegrid(min_pos,max_pos,grid_size,rank,args.world_size)
    xyz=torch.from_numpy(xyz).float().to(rank)
    print(xyz.shape)
    
    # 创建模型并移动到当前GPU
    model = GaussianModel(xyz.shape[0],scale,bin_resolution,num_bins,dataset.t0,decay,confocal,laserOrigin,cameraPos,cameraOrigin).to(rank)

    ddp_model = DDP(model, device_ids=[rank])
    
    # 优化器
    # optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    l = [
            {'params': [model.colours], 'lr': 0.0025, "name": "colours"},
            {'params': [model.pre_act_opacities], 'lr': 0.025, "name": "opacity"},
            {'params': [model.pre_act_scales], 'lr': 0.001, "name": "scaling"}
        ]
    optimizer = torch.optim.Adam(l)
    
    for itr in range(1,args.num_itrs):
        loss=0
        sample_num=1 # sample_num和内存占用成正比，因此可以调小一些

        for iii in range(sample_num):        
            try:
                data = next(train_itr)
            except StopIteration:
                train_itr = iter(train_loader)
                data = next(train_itr)

            laserPos=data["point"].to(rank)
            gt_hist=data["hist"].reshape(-1).to(rank)
            
            optimizer.zero_grad()

            hist = ddp_model(xyz,laserPos)

            # 每个结点计算损失，DDP会自动将梯度all-reduce，实际上每个GPU分别进行了拟合
            loss += torch.mean((hist-gt_hist).abs())

        loss=loss/sample_num
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"[*] Itr: {itr:07d} | Loss: {loss:0.3f} |")

            with torch.no_grad():
                if itr%50==0:  
                    plot_hist(hist,gt_hist,itr)
                
                if itr%500==0:
                    local_params = model.get_all_parameters()
                    rho =local_params["rho"].detach().view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu().numpy()
                    o = local_params["o"].detach().view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu().numpy()
                    scale= local_params["scale"].detach().view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu().numpy()
                    scipy.io.savemat(f"temp/result{itr}.mat",{"rho":rho,"o":o,"scale":scale})
    
    local_params = model.get_all_parameters()
    dic = gather_all_parameters(rank, args.world_size, local_params,pixels)

    if rank == 0:
        scipy.io.savemat("temp/result.mat",dic)
        rho=dic["rho"]
        intensity=np.max(rho,axis=2)
        depth=np.argmax(rho,axis=2)
        intensity=(intensity-np.min(intensity))/(np.max(intensity)-np.min(intensity))
        depth=(depth-np.min(depth))/(np.max(depth)-np.min(depth))
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(depth)
        plt.subplot(1,2,2)
        plt.imshow(intensity)
        plt.savefig("temp/result.png")

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default="shelf_targets_lighton_data/", type=str,
        help="Path to the dataset."
    )
    parser.add_argument(
        "--num_itrs", default=2001, type=int,
        help="Number of iterations to train the model."
    )
    parser.add_argument(
        "--world_size", default=4, type=int,
        help="Number of cuda."
    )
    arg = parser.parse_args()
    
    # 使用多进程启动训练
    mp.spawn(
        train,
        args=(arg,),
        nprocs=arg.world_size,
        join=True
    )