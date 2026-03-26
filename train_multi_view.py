# multi-view reconstruction based on volume rendering

import os
import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
import torch.nn.functional as F
from data_utils import save_ply,save_mat,plot_hist, makegrid

from dataset import MultiViewDataset
import time
import scipy
import matplotlib.pyplot as plt

from gaussian import Gaussians
from scene import Scene

import imageio
from PIL import Image
from tqdm import tqdm
from pytorch3d.renderer.cameras import FoVPerspectiveCameras, look_at_view_transform
import math

### 随机初始化训练模型
def run_training(args):
    torch.manual_seed(16)
    
    # 参数可根据具体情况调整，但一般来说是够用的
    decay=4
    scale=0.002
    thresh=1e-4
    num_itrs=501
    train_fast=False
    ratio=[0.85,0.85,0.85]

    min_pos=[-0.3,-0.3,-0.3] ## frontback_bunny数据参数
    max_pos=[0.3,0.3,0.3]
    grid_size=[0.03,0.03,0.01]
    
    # min_pos=[-0.2,-0.3,-0.2] ## frontback_hydrant数据参数
    # max_pos=[0.2,0.3,0.2]
    # grid_size=[0.03,0.03,0.01]
    
    # min_pos=[-0.3,-0.3,-0.15] ## frontback_christ数据参数
    # max_pos=[0.3,0.3,0.15]
    # grid_size=[0.02,0.02,0.01]

    dataset= MultiViewDataset(args.data_path)
    bin_resolution=dataset.bin_resolution
    device=args.device
    num_bins=dataset.M

    object_center=np.array(max_pos)/2+np.array(min_pos)/2
    radius=np.array(max_pos)/2-np.array(min_pos)/2
    
    # 生成初始方格
    xyz,pixels,grid_size=makegrid(min_pos,max_pos,grid_size)
    print(xyz.shape, pixels)
    
    # 初始化Gaussian
    gaussians = Gaussians(init_type="points", device=args.device, colour_dim=1, scale=scale,means=xyz,use_sigmoid=False)

    start=time.time()

    print("radius:",gaussians.radius)
    print("center:",object_center)
    print("bin resolution:",bin_resolution)
    print("point number:",gaussians.means.shape[0])

    train_loader = DataLoader(
        dataset, batch_size=1,shuffle=True
    )
    train_itr = iter(train_loader)
    
    ### 开始训练
    gaussians.training_setup(train_fast=train_fast)
    
    # 体渲染类
    scene=Scene(gaussians)

    current_step=grid_size[0]

    for itr in range(1,num_itrs):

        sample_num=1

        for iii in range(sample_num):
            try:
                data = next(train_itr)
            except StopIteration:
                train_itr = iter(train_loader)
                data = next(train_itr)

            scan_point=data["point"].flatten().cpu().numpy().tolist()
            gt_hist=data["hist"].reshape(-1).to(device)

            dist=np.sqrt((scan_point[0]-object_center[0])**2+(scan_point[1]-object_center[1])**2+(scan_point[2]-object_center[2])**2) # 扫描点到场景中心的距离
            fov_radius=np.sqrt(radius[0]**2+radius[1]**2+radius[2]**2) # 场景半径
            fov=float(2*np.asin(fov_radius/dist))
            R, T = look_at_view_transform(eye=(scan_point,),at=(object_center.tolist(),),up=((0, 1, 0),)) # 因为高斯元中心在原点，因此at就是原点
            current_camera = FoVPerspectiveCameras(
                znear=0.1,zfar=10.0,
                fov=fov,degrees=False, # radian
                R=R, T=T
            ).to(device)

            img_size=(64,64)
            gaussians_per_splat=-1
            current_camera.image_size=(img_size,)

            # Rendering histogram using gaussian splatting
            hist = scene.render_conf_hist(current_camera,bin_resolution,num_bins,dataset.t0,decay,gaussians_per_splat,img_size)

            # loss=torch.mean((hist-gt_hist).abs())
            loss = F.poisson_nll_loss(hist, gt_hist, log_input=False)
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
    
    # ## 删除在最外一圈记录残差的点
    # prune_mask1=torch.where(torch.abs(gaussians.means[:,0]-object_center[0])>radius[0]*ratio[0], True, False).flatten()
    # prune_mask2=torch.where(torch.abs(gaussians.means[:,1]-object_center[1])>radius[1]*ratio[1], True, False).flatten()
    # prune_mask3=torch.where(torch.abs(gaussians.means[:,2]-object_center[2])>radius[2]*ratio[2], True, False).flatten()
    # prune_mask=torch.logical_or(torch.logical_or(prune_mask1,prune_mask2),prune_mask3)
    # gaussians.prune_points(prune_mask)
    # print(f"prune number: {torch.sum(prune_mask).item()}")

    end=time.time()
    print("Training Completed. Training time:", end-start)

    save_ply("temp/result.ply",gaussians)
    print("Save ply!")
    
    save_mat("temp/result.mat",gaussians,min_pos,[current_step,current_step,grid_size[2]],pixels)
    print("Save mat!")

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default="data/frontback_bunny256.mat", type=str,
        help="Path to the dataset."
    )
    parser.add_argument("--device", default="cuda:1", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    run_training(args)