import os
import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
from data_utils import save_ply,OptimizationParams,plot_hist

from dataset import NLOSDataset
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
    
    scale=0.005 # 默认大小
    ratio=[0.85,0.85,0.85]

    # # 随机初始化
    # radius=0.65 ## cow数据的参数
    # object_center=(0.0,0.0,1.3)
    radius=[0.4,0.4,0.2] ## mannequin数据的参数
    object_center=(0.0,0.0,0.55)
    scale=0.005
    # radius=[0.3,0.3,0.3] ## teapot数据的参数 
    # object_center=(0.0821,0.2270,1.1992)
    # radius=[0.5,0.5,0.4] ## bunny的参数
    # object_center=(0.0037,0.1018,0.8335)
    # scale=0.008
    # radius=[0.95,0.95,0.4] ## fk-dragon数据参数
    # object_center=(-0.25,0.1,1.45)
    # ratio=[0.8,0.8,0.4]

    dataset= NLOSDataset(args.data_path,device=args.device)
    
    bin_resolution=dataset.bin_resolution
    nums_bin=dataset.M
    
    gaussians = Gaussians(
        num_points=10000, init_type="random",
        device=args.device, isotropic=True,
        colour_dim=1,extent=radius,center=object_center,scale=scale
    )

    save_ply("temp/init.ply",gaussians)

    scene = Scene(gaussians)
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

            scan_point=data["point"]
            gt_hist=data["hist"].reshape(-1)

            # Rendering histogram using gaussian splatting
            hist= scene.render_conf_hist(scan_point,bin_resolution,nums_bin)

            loss+=torch.mean((hist-gt_hist).abs())
        
        loss=loss/sample_num
        loss.backward()
        loss_list.append(loss.item())

        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            print(f"[*] Itr: {itr:07d} | Loss: {loss:0.3f} |")

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
            
            plot_hist(hist,gt_hist,itr)

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
    prune_mask1=torch.where(torch.abs(gaussians.means[:,0]-object_center[0])>gaussians.radius*ratio[0], True, False).flatten()
    prune_mask2=torch.where(torch.abs(gaussians.means[:,1]-object_center[1])>gaussians.radius*ratio[1], True, False).flatten()
    prune_mask3=torch.where(torch.abs(gaussians.means[:,2]-object_center[2])>gaussians.radius*ratio[2], True, False).flatten()
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
    parser.add_argument("--device", default="cuda:1", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    run_training(args)