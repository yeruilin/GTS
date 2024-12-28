import os
import torch
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from model import Scene, Gaussians
from torch.utils.data import DataLoader
from data_utils import visualize_renders,save_ply,OptimizationParams
from pytorch3d.renderer.cameras import PerspectiveCameras,FoVPerspectiveCameras, look_at_view_transform
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dataset import ConfocalDataset
import time
import math

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

    dataset= ConfocalDataset(args.data_path,device=args.device)
    img_size=(dataset.N,dataset.N) # 渲染图片大小
    object_center=dataset.obj_center
    radius=dataset.obj_radius
    bin_resolution=dataset.bin_resolution
    nums_bin=dataset.M

    print("radius:",radius)
    print("object center:",object_center)

    train_loader = DataLoader(
        dataset, batch_size=1, shuffle=True
    )
    train_itr = iter(train_loader)

    # Init gaussians and scene
    gaussians = Gaussians(
        init_type="points",load_path="yrl_cow_data/cow_points.mat",
        device=args.device, isotropic=True,
        colour_dim=1,extent=radius
    )

    save_ply("temp/init.ply",gaussians.means,gaussians.colours,gaussians.pre_act_opacities,gaussians.pre_act_scales,gaussians.pre_act_quats,colour_dim=1)

    scene = Scene(gaussians)

    # Making gaussians trainable and setting up optimizer
    make_trainable(gaussians)
    opt_param=OptimizationParams() # 设置优化参数
    opt_param.densification_interval=1000
    opt_param.densify_from_iter=1
    opt_param.densify_grad_threshold=2e-3
    gaussians.training_setup(opt_param) # 设置优化模式

    bg_colour=(0.0,0.0,0.0) # 白色背景

    start=time.time()

    # Training loop
    for itr in range(1,args.num_itrs):

        # Fetching data
        try:
            data = next(train_itr)
        except StopIteration:
            train_itr = iter(train_loader)
            data = next(train_itr)
        
        gaussians.update_learning_rate(itr) # 更新学习率

        gt_hist=data["hist"]
        scan_point=data["point"]
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
        hist,means_2D,radii = scene.render_conf_hist(current_camera,bin_resolution,nums_bin,
                                        args.gaussians_per_splat,img_size,bg_colour,no_grad=False)
        
        visibility_filter= (radii > 0).nonzero() # 选出所有在screen上大小超过0的索引

        hist_max=torch.max(hist)
        print(hist_max)

        # Compute loss
        hist=hist/(hist_max+1e-5)
        loss=torch.mean((hist-gt_hist).abs())
        loss.backward()

        if itr%1000==0:
            save_ply(f"temp/result{itr}.ply",gaussians.means,gaussians.colours,gaussians.pre_act_opacities,gaussians.pre_act_scales,gaussians.pre_act_quats,colour_dim=1)

        ## 自适应更新density
        with torch.no_grad(): 
            if itr < opt_param.densify_until_iter:
                # 记录每个片元在图像上最大的半径
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # 记录每个片元在图像上半径的梯度
                gaussians.add_densification_stats(means_2D, visibility_filter)

                # 一段时间要增加或删减高斯片元
                if itr > opt_param.densify_from_iter and itr % opt_param.densification_interval == 0:
                    print("densify_and_prune")
                    size_threshold = 10 if itr > opt_param.opacity_reset_interval else None # 片元在图片上的大小不超过20
                    gaussians.densify_and_prune(
                        grad_threshold=opt_param.densify_grad_threshold, 
                        min_opacity=0.005, 
                        extent=radius, 
                        max_screen_size=size_threshold, 
                        radii=radii
                    )
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)

        print(f"[*] Itr: {itr:07d} | Loss: {loss:0.3f}")

    end=time.time()
    print("Training Completed. Training time:", end-start)
    # Saving Gaussian primitives (.ply)
    save_ply("temp/result.ply",gaussians.means,gaussians.colours,gaussians.pre_act_opacities,gaussians.pre_act_scales,gaussians.pre_act_quats,colour_dim=1)
    print("Save ply!")

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path", default="./output", type=str,
        help="Path to the directory where output should be saved to."
    )
    parser.add_argument(
        "--data_path", default="gaussian_cow.mat", type=str,
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
