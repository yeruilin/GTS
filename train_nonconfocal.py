import os
import torch
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_utils import save_ply,OptimizationParams,wasserstein_distance,plot_hist

from dataset import NonconfDataset
import time
import scipy
import matplotlib.pyplot as plt

### 整个训练过程分为两步：
### 1.首先按照类似于BP的思路只按照深度来优化color
### 2.随后全面优化scale,opacity等参数

from model2 import Scene, Gaussians

def make_trainable(gaussians):

    ### YOUR CODE HERE ###
    # HINT: You can access and modify parameters from gaussians
    gaussians.means.requires_grad=True
    gaussians.pre_act_scales.requires_grad=True
    gaussians.colours.requires_grad=True
    gaussians.pre_act_opacities.requires_grad=True

### 随机初始化训练模型
def run_training(args):
    torch.manual_seed(16)

    scale=0.005

    # # 随机初始化
    # radius=[0.2,0.2,0.2] ## K的参数
    # object_center=(0,0,0.26)
    # scale=0.002 # 范围太小了，所以需要片元小一点

    # radius=[0.6,0.6,0.6] ## bunny的参数
    # object_center=(0.0037,0.1018,0.8335)
    # scale=0.015
    # radius=[1.0,0.6,1.0] ## phasor_id3的参数
    # object_center=(-0.20,0.05,1.40)
    radius=[0.825,0.75,0.25] ## phasor_id5的参数
    object_center=(-0.625,1.25,0.95)

    gaussians = Gaussians(
        num_points=25000, init_type="random",
        device=args.device, isotropic=True,
        colour_dim=1,extent=radius,center=object_center,scale=scale
    )

    save_ply("temp/init.ply",gaussians)

    scene = Scene(gaussians)
    start=time.time()
    
    dataset= NonconfDataset(args.data_path,device=args.device)
    bin_resolution=dataset.bin_resolution
    nums_bin=dataset.M

    print("radius:",radius)
    print("center:",object_center)
    print("bin resolution:",bin_resolution)
    print("laserOrigin:",dataset.laserOrigin)
    print("cameraOrigin:",dataset.cameraOrigin)
    print("cameraPos:",dataset.cameraPos)
    print("t0:",dataset.t0)

    train_loader = DataLoader(
        dataset, batch_size=1,shuffle=True
    )
    train_itr = iter(train_loader)
    
    ### 开始训练
    # 阶段一：清掉无用位置的点
    opt_param=OptimizationParams()
    gaussians.training_setup(opt_param)
    make_trainable(gaussians)

    loss_list=[]

    for itr in range(1,args.num_itrs):
        loss=0
        sample_num=16

        for iii in range(sample_num):
            try:
                data = next(train_itr)
            except StopIteration:
                train_itr = iter(train_loader)
                data = next(train_itr)
            laserPos=data["point"]
            gt_hist=data["hist"].reshape(-1)

            # Rendering histogram using gaussian splatting
            hist= scene.render_nonconf_hist(laserPos,dataset.laserOrigin,dataset.cameraPos,dataset.cameraOrigin,bin_resolution,nums_bin,dataset.t0)

            loss+=torch.mean((hist-gt_hist).abs())
        
        loss=loss/sample_num
        loss.backward()
        loss_list.append(loss.item())
            
        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            print(f"[*] Itr: {itr:07d} | Loss: {loss:0.3f} |")

        if itr%10==0:  
            plot_hist(hist,gt_hist,itr)   

        if itr%50==0:
            # scipy.io.savemat(f"temp/hist{itr}.mat",{"hist":,"gt_hist":gt_hist.detach().cpu().numpy()})
            # 防止出现太大的片元
            select_mask=torch.where(gaussians.get_scaling[:,0]>0.02, True, False).flatten()
            gaussians.density_and_split1(select_mask,copy_num=1)
            print(f"split number: {torch.sum(select_mask).item()}")

            # 直接删除太大的片元
            prune_mask=torch.where(gaussians.get_scaling[:,0]>0.03, True, False).flatten()
            gaussians.prune_points(prune_mask)
            print(f"prune number: {torch.sum(prune_mask).item()}")

            save_ply(f"temp/result{itr}.ply",gaussians)

        if itr==200 or itr%500==0:
            prune_mask=torch.where(gaussians.get_colour<=1e-4, True, False).flatten()
            gaussians.prune_points(prune_mask)
            print(f"prune number: {torch.sum(prune_mask).item()}")
        
        if itr==200:
            gaussians.densify_and_clone1(copy_num=2,std_multiple=3)
            
        if itr==501:
            gaussians.densify_and_clone1(copy_num=2,std_multiple=5)
        
        # # 在上面的裁剪策略几乎无效的时候，可以把低于均值的位置裁掉，高于均值的进行拷贝
        # if itr==1000:
        #     # 删除小于均值的位置
        #     if thresh==0:
        #         prune_mask=torch.where(gaussians.get_colour<=torch.mean(gaussians.get_colour), True, False).flatten()
        #     else:
        #         prune_mask=torch.where(gaussians.get_colour<=thresh, True, False).flatten()
            
        #     # 删除在最外一圈记录残差的点
        #     ratio=0.85
        #     prune_mask1=torch.where(torch.abs(gaussians.means[:,0]-object_center[0])>radius*ratio, True, False).flatten()
        #     prune_mask2=torch.where(torch.abs(gaussians.means[:,1]-object_center[1])>radius*ratio, True, False).flatten()
        #     prune_mask3=torch.where(torch.abs(gaussians.means[:,2]-object_center[2])>radius*ratio, True, False).flatten()
        #     prune_mask_=torch.logical_or(torch.logical_or(prune_mask1,prune_mask2),prune_mask3)
        #     prune_mask=torch.logical_or(prune_mask,prune_mask_)

        #     gaussians.prune_points(prune_mask)
        #     print(f"prune number: {torch.sum(prune_mask).item()}")

        #     # 剩下的点全都进行拷贝
        #     gaussians.densify_and_clone1(copy_num=2)
        #     print(f"Gaussian number left: {gaussians.means.shape[0]}")
    
    # 删除在最外一圈记录残差的点
    ratio=0.85
    prune_mask1=torch.where(torch.abs(gaussians.means[:,0]-object_center[0])>radius[0]*ratio, True, False).flatten()
    prune_mask2=torch.where(torch.abs(gaussians.means[:,1]-object_center[1])>radius[1]*ratio, True, False).flatten()
    prune_mask3=torch.where(torch.abs(gaussians.means[:,2]-object_center[2])>radius[2]*ratio, True, False).flatten()
    prune_mask=torch.logical_or(torch.logical_or(prune_mask1,prune_mask2),prune_mask3)
    gaussians.prune_points(prune_mask)
    print(f"prune number: {torch.sum(prune_mask).item()}")

    end=time.time()
    print("Training Completed. Training time:", end-start)

    save_ply("temp/result.ply",gaussians)
    print("Save ply!")

    plt.plot(loss_list)
    plt.savefig("temp/loss_figure.png")
    print("Save loss figure!")

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default="data/lct_mannequin.mat", type=str, # "yrl_cow_data/cow.mat"
        help="Path to the dataset."
    )
    parser.add_argument(
        "--gaussians_per_splat", default=-1, type=int,
        help=(
            "Number of gaussians to splat in one function call. If set to -1, "
            "then all gaussians in the scene are splat in a single function call. "
            "If set to any other positive interger, then it determines the number of "
            "gaussians to splat per function call (the last function call might splat "
            "lesser number of gaussians). In general, the algorithm can run faster "
            "if more gaussians are splat per function call, but at the cost of higher GPU "
            "memory consumption."
        )
    )
    parser.add_argument(
        "--num_itrs", default=501, type=int,
        help="Number of iterations to train the model."
    )
    parser.add_argument("--device", default="cuda:0", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    run_training(args)