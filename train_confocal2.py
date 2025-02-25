import os
import torch
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_utils import save_ply,OptimizationParams,wasserstein_distance,plot_hist

from dataset import NLOSDataset
import time
import scipy
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

### 整个训练过程分为两步：
### 1.首先按照类似于BP的思路只按照深度来优化color
### 2.随后全面优化scale,opacity等参数

# 本代码处理第二阶段（泼墨优化）

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

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    # path="test_bunny/result3000.ply"
    path="temp/result1500.ply"

    gaussians = Gaussians(
        init_type="gaussians",load_path=path,
        device=args.device,colour_dim=1
    )

    ## 初始一个比较大的规模然后开始缩小
    gaussians.pre_act_scales[:,:]=-4.6

    radius=gaussians.radius.item()
    object_center=gaussians.center.cpu().numpy()

    save_ply("temp/init.ply",gaussians)

    scene = Scene(gaussians)
    start=time.time()
    
    dataset= NLOSDataset(args.data_path,device=args.device)

    bin_resolution=dataset.bin_resolution
    nums_bin=dataset.M

    print("radius:",radius)
    print("center:",object_center)
    print("bin resolution:",bin_resolution)
    print("width:",dataset.width)

    train_loader = DataLoader(
        dataset, batch_size=1,shuffle=True
    )
    train_itr = iter(train_loader)
    
    ### 开始训练
    opt_param=OptimizationParams()
    opt_param.densification_interval=50
    opt_param.densify_from_iter=1
    opt_param.position_lr_init=0.0001
    gaussians.training_setup(opt_param)
    make_trainable(gaussians)

    loss_list=[]

    # Training loop
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

            hist= scene.render_conf_hist(scan_point,bin_resolution,nums_bin)

            loss+=torch.mean((hist-gt_hist).abs())
        
        loss=loss/sample_num
        loss.backward()
        loss_list.append(loss.item())

        print(torch.max(gaussians.means.grad),torch.mean(gaussians.means.grad))

        if itr==100:
            # 对于尺寸大的点，需要强行设置比较小的scale
            prune_mask=torch.where(gaussians.get_scaling[:,0]>0.03, True, False).flatten()
            gaussians.prune_points(prune_mask)
            # gaussians.set_scale(prune_mask,0.01)
            print(f"prune number: {torch.sum(prune_mask).item()}")

        if itr%50==0:
            save_ply(f"temp/splat_result{itr}.ply",gaussians)
            # scipy.io.savemat(f"temp/hist{itr}.mat",{"hist":hist.detach().cpu().numpy(),"gt_hist":gt_hist.detach().cpu().numpy()})
            plot_hist(hist,gt_hist,itr)

            ## 把比较大的片元分成小片元
            if itr>100:
                select_mask=torch.where(gaussians.get_scaling[:,0]>0.02, True, False).flatten()
                gaussians.density_and_split1(select_mask,copy_num=2)
                print(f"split number: {torch.sum(select_mask).item()}")

        # print(torch.max(gaussians.pre_act_scales.grad),torch.mean(gaussians.pre_act_scales.grad))

        # with torch.no_grad():
        #     # # 裁剪深度不在指定范围的片元
        #     if True:
        #         # 统计梯度
        #         if itr > opt_param.densify_from_iter:
        #             visibility_filter=torch.ones(gaussians.means.shape[0], dtype=torch.bool).to(args.device) # 全都记录梯度
        #             gaussians.add_densification_stats(gaussians.means, visibility_filter)
                    
        #         # 一段时间要增加或删减高斯片元
        #         if itr > opt_param.densify_from_iter and itr % opt_param.densification_interval == 0:
        #             print("densify_and_prune")
        #             gaussians.densify_and_prune(
        #                 grad_threshold=densify_grad_threshold, 
        #                 min_opacity=0.1, 
        #                 extent=radius
        #             )

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none = True)
        print(f"[*] Itr: {itr:07d} | Loss: {loss:0.3f} |")


    end=time.time()
    print("Training Completed. Training time:", end-start)
    # Saving Gaussian primitives (.ply)
    save_ply("temp/result.ply",gaussians)
    print("Save ply!")

    plt.plot(loss_list)
    plt.savefig("temp/loss_figure.png")
    print("Save loss figure!")

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path", default="./output", type=str,
        help="Path to the directory where output should be saved to."
    )
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
        "--num_itrs", default=1001, type=int,
        help="Number of iterations to train the model."
    )
    parser.add_argument(
        "--viz_freq", default=20, type=int,
        help="Frequency with which visualization should be performed."
    )
    parser.add_argument("--device", default="cuda:0", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    run_training(args)