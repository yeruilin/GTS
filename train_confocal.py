import os
import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
import torch.nn.functional as F
from data_utils import save_ply,save_mat,plot_hist, makegrid

from dataset import NLOSDataset
import time
import scipy
import matplotlib.pyplot as plt

from gaussian import Gaussians

### 用八叉树策略来增删椭球

### 随机初始化训练模型
def run_training(args):
    torch.manual_seed(16)
    
    scale=0.002 # 默认大小
    ratio=[0.85,0.85,0.85]
    use_filter=False # 光子太少，波形不连续需要先平滑
    num_itrs=501
    decay=4 # 衰减系数
    train_fast=True
    thresh=1e-4 

    # 初始化
    # min_pos=[-0.5,-0.5,0.45] ## bunny的参数
    # max_pos=[0.5,0.5,1.15]
    # grid_size=[0.03,0.03,0.01]

    # min_pos=[-1.0,-1.0,1.0] ## fk_teaser180_clip数据参数
    # max_pos=[1.0,1.0,1.9]
    # grid_size=[0.03,0.03,0.005]
    # train_fast=False
    # num_itrs=1001

    min_pos=[-1.0,-1.0,1.2] ## fk-dragon数据参数
    max_pos=[1.0,1.0,1.5]
    grid_size=[0.04,0.04,0.01]
    thresh=5e-4
    # filter=True
    # num_itrs=3001 # 信噪比越低，所需轮次越大

    # min_pos=[-0.7,-0.7,0.9] ## fk-statue数据参数
    # max_pos=[0.7,0.7,1.3]
    # grid_size=[0.04,0.04,0.01]

    # min_pos=[-0.3,-0.3,0.3]
    # max_pos=[0.3,0.3,0.7] ## mannequin数据的参数
    # grid_size=[0.04,0.04,0.005]
    
    object_center=np.array(max_pos)/2+np.array(min_pos)/2 ## front_bunny的参数
    radius=np.array(max_pos)/2-np.array(min_pos)/2

    dataset= NLOSDataset(args.data_path,filter=use_filter)
    device=args.device
    bin_resolution=dataset.bin_resolution
    nums_bin=dataset.M
    
    # 生成初始方格
    xyz,pixels,grid_size=makegrid(min_pos,max_pos,grid_size)
    print(xyz.shape, pixels)
    
    gaussians = Gaussians(init_type="points", device=args.device, colour_dim=1, scale=scale,means=xyz)

    start=time.time()

    print("radius:",gaussians.radius)
    print("center:",object_center)
    print("bin resolution:",bin_resolution)
    print("width:",dataset.width)
    print("point number:",gaussians.means.shape[0])

    train_loader = DataLoader(
        dataset, batch_size=1,shuffle=True
    )
    train_itr = iter(train_loader)
    
    ### 开始训练
    gaussians.training_setup(train_fast=train_fast)

    current_step=grid_size[0]

    for itr in range(1,num_itrs):
        loss=0
        sample_num=64 # 提高效果可以继续提高

        for iii in range(sample_num):
            try:
                data = next(train_itr)
            except StopIteration:
                train_itr = iter(train_loader)
                data = next(train_itr)

            scan_point=data["point"].to(device)
            gt_hist=data["hist"].reshape(-1).to(device)

            # Rendering histogram using gaussian splatting
            hist= gaussians.render_conf_hist2(scan_point,bin_resolution,nums_bin,dataset.t0,decay)

            # loss=torch.mean((hist-gt_hist).abs())
            loss = F.poisson_nll_loss(hist, gt_hist, log_input=False) # poisson loss, 高信噪比特别好用
            loss=loss/sample_num
            
            loss.backward()
        
        # gradient accumulation
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none = True)
        gaussians.scheduler.step()

        with torch.no_grad():
            
            print(f"[*] Itr: {itr:07d} | Loss: {loss:0.3f} |")

            if itr%50==0:
                # scipy.io.savemat(f"temp/hist{itr}.mat",{"hist":,"gt_hist":gt_hist.detach().cpu().numpy()})
                # 防止出现太大的片元
                select_mask=torch.where(gaussians.get_scaling[:,0]>0.03, True, False).flatten()
                gaussians.density_and_split1(select_mask,copy_num=1)
                print(f"split number: {torch.sum(select_mask).item()}")

                # 直接删除太大的片元
                prune_mask=torch.where(gaussians.get_scaling[:,0]>0.04, True, False).flatten()
                gaussians.prune_points(prune_mask)

                save_ply(f"temp/result{itr}.ply",gaussians)

            if itr%50==0:
                plot_hist(hist,gt_hist,itr)

            if itr==200 or itr%500==0:
                # 删除颜色过小的片元
                prune_mask=torch.where(gaussians.get_colour<=thresh, True, False).flatten()
                gaussians.prune_points(prune_mask)

            if itr%500==0:
                save_mat(f"temp/result{itr}.mat",gaussians,min_pos,[current_step,current_step,grid_size[2]],pixels)
            
            if itr==200:
                current_step=current_step/2
                pixels[0]=pixels[0]*2
                pixels[1]=pixels[1]*2
                
                # 扩大有物体的位置
                clone_mask=torch.where(gaussians.get_colour>thresh, True, False).flatten()
                gaussians.densify_and_clone2(clone_mask,current_step)     
    
    ## 删除在最外一圈记录残差的点
    prune_mask1=torch.where(torch.abs(gaussians.means[:,0]-object_center[0])>radius[0]*ratio[0], True, False).flatten()
    prune_mask2=torch.where(torch.abs(gaussians.means[:,1]-object_center[1])>radius[1]*ratio[1], True, False).flatten()
    prune_mask3=torch.where(torch.abs(gaussians.means[:,2]-object_center[2])>radius[2]*ratio[2], True, False).flatten()
    prune_mask=torch.logical_or(torch.logical_or(prune_mask1,prune_mask2),prune_mask3)
    gaussians.prune_points(prune_mask)

    end=time.time()
    print("Training Completed. Training time:", end-start)

    save_ply("temp/result.ply",gaussians)
    print("Save ply!")
    
    save_mat("temp/result.mat",gaussians,min_pos,[current_step,current_step,grid_size[2]],pixels)
    print("Save mat!")

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default="data/fk_dragon180.mat", type=str, # "yrl_cow_data/cow.mat"
        help="Path to the dataset."
    )
    parser.add_argument("--device", default="cuda:1", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    run_training(args)