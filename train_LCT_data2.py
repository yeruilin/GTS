import os
import torch
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from model import Scene, Gaussians
from torch.utils.data import DataLoader
from data_utils import save_ply,OptimizationParams,wasserstein_distance
from pytorch3d.renderer.cameras import PerspectiveCameras,FoVPerspectiveCameras, look_at_view_transform

from dataset import LCTDataset,ConfocalDataset
import time
import math
import scipy
import matplotlib.pyplot as plt

def make_trainable(gaussians):

    ### YOUR CODE HERE ###
    # HINT: You can access and modify parameters from gaussians
    gaussians.means.requires_grad=True
    gaussians.pre_act_scales.requires_grad=True
    gaussians.colours.requires_grad=True
    gaussians.pre_act_opacities.requires_grad=True
    
    if not gaussians.is_isotropic:
        gaussians.pre_act_quats.requires_grad=True

def run_training(args):

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    dataset= ConfocalDataset(args.data_path,device=args.device,is_train=True)
    img_size=(dataset.N,dataset.N) # 渲染图片大小
    bin_resolution=dataset.bin_resolution
    nums_bin=dataset.M

    train_loader = DataLoader(
        dataset, batch_size=1, shuffle=True
    )
    train_itr = iter(train_loader)

    point_path=args.data_path.replace(".mat","_points.mat")

    # Init gaussians and scene
    gaussians = Gaussians(
        init_type="points",load_path=point_path,
        device=args.device, isotropic=True,colour_dim=1
    )
    object_center=gaussians.center.detach().cpu().numpy()
    object_center=(object_center[0],object_center[1],object_center[2])
    radius=gaussians.radius.item()

    # # 随机初始化
    # radius=0.8
    # object_center=(0.0,0.0,1.3)
    # gaussians = Gaussians(
    #     num_points=3000, init_type="random",
    #     device=args.device, isotropic=True,
    #     colour_dim=1,extent=radius
    # )

    print("radius:",radius)
    print("center:",object_center)
    print("bin resolution:",bin_resolution)
    print("width:",dataset.width)

    save_ply("temp/init.ply",gaussians.means,gaussians.colours,gaussians.pre_act_opacities,gaussians.pre_act_scales,gaussians.pre_act_quats,colour_dim=1)

    scene = Scene(gaussians)

    # Making gaussians trainable and setting up optimizer
    make_trainable(gaussians)
    opt_param=OptimizationParams() # 设置优化参数
    opt_param.densification_interval=100 # 进行增删片元的间隔
    opt_param.densify_from_iter=300
    opt_param.densify_grad_threshold=1 # cow的阈值为1，mannequin为10
    # opt_param.position_lr_init=0.00032
    gaussians.training_setup(opt_param) # 设置优化模式

    loss_list=[]

    start=time.time()

    # Training loop
    for itr in range(1,args.num_itrs):
        gaussians.update_learning_rate(itr) # 更新学习率

        l1=0
        lregular=0
        for iii in range(16):
            try:
                data = next(train_itr)
            except StopIteration:
                train_itr = iter(train_loader)
                data = next(train_itr)
            scan_point=data["point"]
            gt_hist=data["hist"]
            dist=math.sqrt((scan_point[0]-object_center[0])**2+(scan_point[1]-object_center[1])**2+(scan_point[2]-object_center[2])**2) # 扫描点到场景中心的距离
            fov=2*math.asin(radius/dist)
            R, T = look_at_view_transform(eye=(scan_point,),at=(object_center,),up=((0, 1, 0),)) # 因为高斯元中心在原点，因此at就是原点
            current_camera = FoVPerspectiveCameras(
                znear=0.1,zfar=10.0,
                fov=fov,degrees=False, # radian
                R=R, T=T
            ).to(args.device)
            current_camera.image_size=(img_size,)

            # Rendering histogram using gaussian splatting
            hist= scene.render_conf_hist(current_camera,bin_resolution,nums_bin,args.gaussians_per_splat,img_size,is_train=True)

            hist_max=torch.max(hist)
            print(hist_max)
            if itr<200:
                l1+=torch.mean((hist-gt_hist).abs())
            else:
                l1+=wasserstein_distance(hist,gt_hist)
        
        loss=l1
        loss.backward()
        loss_list.append(loss.item()) 

        print(torch.max(gaussians.means.grad),torch.mean(gaussians.means.grad))

        if itr%50==0:
            save_ply(f"temp/result{itr}.ply",gaussians.means,gaussians.colours,gaussians.pre_act_opacities,gaussians.pre_act_scales,gaussians.pre_act_quats,colour_dim=1)
            scipy.io.savemat(f"temp/hist{itr}.mat",{"hist":hist.detach().cpu().numpy(),"gt_hist":gt_hist.detach().cpu().numpy()})

        with torch.no_grad():
            # 统计梯度
            if itr > opt_param.densify_from_iter:
                # 记录每个片元在图像上半径的梯度
                visibility_filter=torch.ones(gaussians.means.shape[0], dtype=torch.bool).to(args.device) # 全都记录梯度
                gaussians.add_densification_stats(gaussians.means, visibility_filter)
                
            # 一段时间要增加或删减高斯片元
            if itr > opt_param.densify_from_iter and itr % opt_param.densification_interval == 0:
                print("densify_and_prune")
                gaussians.densify_and_prune(
                    grad_threshold=opt_param.densify_grad_threshold, 
                    min_opacity=0.005, 
                    extent=radius
                )
                save_ply(f"temp/result{itr}_prune.ply",gaussians.means,gaussians.colours,gaussians.pre_act_opacities,gaussians.pre_act_scales,gaussians.pre_act_quats,colour_dim=1)

            ### 一段时间要重置一次透明度，这样可以消除floaters悬浮物
            # # if itr % opt_param.opacity_reset_interval == 0 or (itr == opt_param.densify_from_iter):
            #     print("reset_opacity")
            #     gaussians.reset_opacity(opacity_thresh=0.025)

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            print(f"[*] Itr: {itr:07d} | Loss: {loss:0.3f}")

    end=time.time()
    print("Training Completed. Training time:", end-start)
    # Saving Gaussian primitives (.ply)
    save_ply("temp/result.ply",gaussians.means,gaussians.colours,gaussians.pre_act_opacities,gaussians.pre_act_scales,gaussians.pre_act_quats,colour_dim=1)
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
        "--num_itrs", default=1000, type=int,
        help="Number of iterations to train the model."
    )
    parser.add_argument(
        "--viz_freq", default=20, type=int,
        help="Frequency with which visualization should be performed."
    )
    parser.add_argument("--device", default="cuda:1", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    run_training(args)
