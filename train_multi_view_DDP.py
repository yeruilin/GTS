### 多视角重建

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import matplotlib.pyplot as plt
import scipy
import argparse
from dataset import *
from data_utils import plot_hist

import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

from model import *

def train(rank, args):
    # init DDP
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12345',
        rank=rank,
        world_size=args.world_size
    )

    confocal=True
    decay=4
    scale=0.002
    num_itrs=501
    train_fast=True

    # min_pos=[-0.3,-0.3,-0.3] ## frontback_bunny数据参数
    # max_pos=[0.3,0.3,0.3]
    # grid_size=[0.003,0.003,0.005]
    # view_num=4

    # min_pos=[-0.15,-0.3,-0.3] ## frontback_lion数据参数
    # max_pos=[0.15,0.3,0.3]
    # grid_size=[0.0024,0.0024,0.005]
    # view_num=4
    # num_itrs=2001

    min_pos=[-0.3,-0.3,-0.3] ## frontback_cylinder数据参数
    max_pos=[0.3,0.3,0.3]
    grid_size=[0.003,0.003,0.01]
    view_num=3
    train_fast=False

    dataset= MultiViewDataset(args.data_path)
    bin_resolution=dataset.bin_resolution

    num_bins=dataset.M

    if rank==0:
        print("min_pos:",min_pos)
        print("max_pos:",max_pos)
        print("grid_size:",grid_size)
        print("bin resolution:",bin_resolution)
        print("t0:",dataset.t0)
    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler)
    train_itr = iter(train_loader)
    
    # 生成当前GPU上的高斯中心
    xyz,pixels=makegrid(min_pos,max_pos,grid_size,rank,args.world_size)
    xyz=torch.from_numpy(xyz).float().to(rank)
    print(xyz.shape)
    
    # 创建模型并移动到当前GPU
    model = GaussianModel(xyz.shape[0],scale,bin_resolution,num_bins,dataset.t0,decay,confocal,view_num=view_num).to(rank)

    ddp_model = DDP(model, device_ids=[rank])
    
    # 优化器
    # optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    if train_fast:
        l = [
            {'params': [model.colours], 'lr': 0.0025, "name": "colours"},
            {'params': [model.coefficients], 'lr': 0.02, "name": "coefficient"},
            {'params': [model.opacities], 'lr': 0.02, "name": "opacity"},
            {'params': [model.scales], 'lr': 0.002, "name": "scaling"}
        ]
    else:
        l = [
                {'params': [model.colours], 'lr': 0.001, "name": "colours"},
                {'params': [model.coefficients], 'lr': 0.01, "name": "coefficients"},
                {'params': [model.opacities], 'lr': 0.01, "name": "opacity"},
                {'params': [model.scales], 'lr': 0.001, "name": "scaling"}
            ]
    
    optimizer = torch.optim.Adam(l)
    
    
    for itr in range(1,num_itrs):
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
            view_id=data["view_id"]
            
            optimizer.zero_grad()

            hist = ddp_model(xyz,laserPos,view_id)

            # 每个结点计算损失，DDP会自动将梯度all-reduce，实际上每个GPU分别进行了拟合
            loss += torch.mean((hist-gt_hist).abs())

        loss=loss/sample_num
        loss.backward()
        optimizer.step()
        scheduler.step()

        if rank == 0:
            print(f"[*] Itr: {itr:07d} | Loss: {loss:0.3f} |")

            with torch.no_grad():
                if itr%50==0:  
                    plot_hist(hist,gt_hist,itr)
                
                if itr%500==0:
                    local_params = model.get_all_parameters()
                    rho =local_params["rho"].detach().view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu().numpy()
                    o = local_params["o"].detach().view(pixels[0]//2,pixels[1]//2,pixels[2],view_num).cpu().numpy()
                    c= local_params["c"].detach().view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu().numpy()
                    scale= local_params["scale"].detach().view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu().numpy()
                    scipy.io.savemat(f"temp/result{itr}.mat",{"rho":rho,"opacity":o,"c":c,"scale":scale})
    
    if rank == 0:
        # scipy.io.savemat("temp/result.mat",dic)
        # rho=dic["rho"]
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
        "--data_path", default="data/frontback_cylinder.mat", type=str,
        help="Path to the dataset."
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