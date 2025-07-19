import os
import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
from data_utils import save_ply,OptimizationParams,plot_hist

from dataset import MultiViewDataset
import time
import scipy
import matplotlib.pyplot as plt

from gaussian import Gaussians

### 随机初始化训练模型
def run_training(args):
    torch.manual_seed(16)
    
    confocal=True
    decay=4
    scale=0.1
    num_itrs=501
    train_fast=False
    ratio=[0.85,0.85,0.85]

    # min_pos=[-0.3,-0.3,-0.3] ## frontback_bunny数据参数
    # max_pos=[0.3,0.3,0.3]
    # grid_size=[0.003,0.003,0.005]
    # view_num=2

    # min_pos=[-0.15,-0.3,-0.3] ## frontback_lion数据参数
    # max_pos=[0.15,0.3,0.3]
    # grid_size=[0.0024,0.0024,0.005]
    # view_num=4
    # num_itrs=2001

    # min_pos=[-0.3,-0.3,-0.3] ## frontback_cylinder数据参数
    # max_pos=[0.3,0.3,0.3]
    # grid_size=[0.003,0.003,0.01]
    # view_num=3
    # train_fast=False

    min_pos=[-0.3,-0.3,-0.15] ## frontback_christ数据参数
    max_pos=[0.3,0.3,0.15]
    grid_size=[0.005,0.005,0.005]
    view_num=2

    dataset= MultiViewDataset(args.data_path)
    bin_resolution=dataset.bin_resolution
    device=args.device
    num_bins=dataset.M

    object_center=np.array(max_pos)/2+np.array(min_pos)/2
    radius=np.array(max_pos)/2-np.array(min_pos)/2
    
    gaussians = Gaussians(
        num_points=15000, init_type="random",
        device=args.device, isotropic=True,
        colour_dim=1,extent=radius,center=object_center,scale=scale,view_num=view_num
    )

    save_ply("temp/init.ply",gaussians)

    start=time.time()

    train_loader = DataLoader(dataset, batch_size=1,shuffle=True)
    train_itr = iter(train_loader)
    
    ### 开始训练
    gaussians.training_setup(train_fast=True)

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

            scan_point=data["point"].to(device)
            gt_hist=data["hist"].reshape(-1).to(device)
            view_id=data["view_id"]

            hist= gaussians.render_conf_hist2(scan_point,bin_resolution,num_bins,dataset.t0,decay,view_id)

            loss+=torch.mean((hist-gt_hist).abs())
        
        loss=loss/sample_num
        loss.backward()
        loss_list.append(loss.item())

        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.scheduler.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            print(f"[*] Itr: {itr:07d} | Loss: {loss:0.3f} |")

            if itr%50==0:
                # scipy.io.savemat(f"temp/hist{itr}.mat",{"hist":,"gt_hist":gt_hist.detach().cpu().numpy()})
                # 防止出现太大的片元
                select_mask=torch.where(gaussians.get_scaling[:,0]>scale*1.5, True, False).flatten()
                gaussians.density_and_split1(select_mask,copy_num=1)
                print(f"split number: {torch.sum(select_mask).item()}")

                # 直接删除太大的片元
                prune_mask=torch.where(gaussians.get_scaling[:,0]>scale*2, True, False).flatten()
                gaussians.prune_points(prune_mask)
                print(f"prune number: {torch.sum(prune_mask).item()}")

                save_ply(f"temp/result{itr}.ply",gaussians)

            if itr%50==0:
                plot_hist(hist,gt_hist,itr)

            if itr==200 or itr%500==0:
                prune_mask=torch.where(gaussians.get_colour<=1e-4, True, False).flatten()
                gaussians.prune_points(prune_mask)
                print(f"prune number: {torch.sum(prune_mask).item()}")
            
            if itr==200:
                gaussians.densify_and_clone1(copy_num=2,std_multiple=3)
                
            if itr==501:
                gaussians.densify_and_clone1(copy_num=2,std_multiple=5)
    
    ## 删除在最外一圈记录残差的点
    prune_mask1=torch.where(torch.abs(gaussians.means[:,0]-object_center[0])>radius[0]*ratio[0], True, False).flatten()
    prune_mask2=torch.where(torch.abs(gaussians.means[:,1]-object_center[1])>radius[1]*ratio[1], True, False).flatten()
    prune_mask3=torch.where(torch.abs(gaussians.means[:,2]-object_center[2])>radius[2]*ratio[2], True, False).flatten()
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
        "--data_path", default="data/frontback_christ.mat", type=str,
        help="Path to the dataset."
    )
    parser.add_argument(
        "--gaussians_per_splat", default=4096, type=int,
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