import os
import torch
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader,WeightedRandomSampler
from data_utils import save_ply,OptimizationParams,wasserstein_distance,TVLoss
from pytorch3d.renderer.cameras import PerspectiveCameras,FoVPerspectiveCameras, look_at_view_transform

from dataset import LCTDataset,ConfocalDataset
import time
import math
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
    
    if not gaussians.is_isotropic:
        gaussians.pre_act_quats.requires_grad=True


### 随机初始化训练模型
def run_training(args):
    torch.manual_seed(16)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    # # 随机初始化
    # radius=0.65 ## cow数据的参数
    # object_center=(0.0,0.0,1.3)
    radius=0.25 ## mannequin数据的参数
    object_center=(0.0,0.0,0.6)
    gaussians = Gaussians(
        num_points=5000, init_type="random",
        device=args.device, isotropic=True,
        colour_dim=1,extent=radius,center=object_center
    )

    fov_radius=1.2*radius

    save_ply("temp/init.ply",gaussians.means,gaussians.colours,gaussians.pre_act_opacities,gaussians.pre_act_scales,gaussians.pre_act_quats,colour_dim=1)

    scene = Scene(gaussians)
    start=time.time()
    
    start_index=150 # 参数是mannequin的
    end_index=350
    dataset= ConfocalDataset(args.data_path,device=args.device,is_train=True,start_index=start_index,end_index=end_index) 
    img_size=(dataset.N,dataset.N) # 渲染图片大小
    bin_resolution=dataset.bin_resolution
    nums_bin=dataset.M

    print("radius:",radius)
    print("center:",object_center)
    print("bin resolution:",bin_resolution)
    print("width:",dataset.width)

    sampler = WeightedRandomSampler(dataset.weights, len(dataset), replacement=True)
    indices = torch.randperm(dataset.M)

    train_loader = DataLoader(
        dataset, batch_size=1,shuffle=True#, sampler=sampler
    )
    train_itr = iter(train_loader)

    # z_mask=torch.ones((gaussians.means.shape[0],),dtype=torch.bool,device=args.device) # 设置深度过滤器
    # # 对随机初始化结果进行裁剪
    # for itr in range(200):
    #     try:
    #         data = next(train_itr)
    #     except StopIteration:
    #         train_itr = iter(train_loader)
    #         data = next(train_itr)
    #     scan_point=data["point"]
    #     gt_hist=data["hist"]
    #     z1,z2=data["z_range"]
    #     z1=z1.to(args.device)
    #     z2=z2.to(args.device)
    #     dist=math.sqrt((scan_point[0]-object_center[0])**2+(scan_point[1]-object_center[1])**2+(scan_point[2]-object_center[2])**2) # 扫描点到场景中心的距离
    #     fov=2*math.asin(fov_radius/dist)
    #     R, T = look_at_view_transform(eye=(scan_point,),at=(object_center,),up=((0, 1, 0),)) # 因为高斯元中心在原点，因此at就是原点
    #     current_camera = FoVPerspectiveCameras(
    #         znear=0.1,zfar=10.0,
    #         fov=fov,degrees=False, # radian
    #         R=R, T=T
    #     ).to(args.device)
    #     current_camera.image_size=(img_size,)

    #     # Rendering histogram using gaussian splatting
    #     hist,z_vals= scene.render_conf_hist(current_camera,bin_resolution,nums_bin,args.gaussians_per_splat,img_size,is_train=True)
    #     # filtering the gaussian outside the z range
    #     z_mask=torch.logical_and(z_mask,torch.logical_and(z_vals<z2,z_vals>z1))

    # print(f"depth prune number: {torch.sum(~z_mask).item()}")
    # gaussians.prune_points_simple(~z_mask)
    
    ### 开始训练
    # Making gaussians trainable and setting up optimizer
    # make_trainable(gaussians)
    opt_param=OptimizationParams()
    gaussians.training_setup(opt_param)
    gaussians.colours.requires_grad=True

    loss_list=[]

    depth_prune_itr=200

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
            z1,z2=data["z_range"]
            z1=z1.to(args.device)
            z2=z2.to(args.device)
            dist=math.sqrt((scan_point[0]-object_center[0])**2+(scan_point[1]-object_center[1])**2+(scan_point[2]-object_center[2])**2) # 扫描点到场景中心的距离
            fov=2*math.asin(fov_radius/dist)
            R, T = look_at_view_transform(eye=(scan_point,),at=(object_center,),up=((0, 1, 0),)) # 因为高斯元中心在原点，因此at就是原点
            current_camera = FoVPerspectiveCameras(
                znear=0.1,zfar=10.0,
                fov=fov,degrees=False, # radian
                R=R, T=T
            ).to(args.device)
            current_camera.image_size=(img_size,)

            # Rendering histogram using gaussian splatting
            hist,z_vals= scene.render_conf_hist(current_camera,bin_resolution,nums_bin,args.gaussians_per_splat,img_size,is_train=True)

            if itr<depth_prune_itr: # 最开始和reset opacity并删除无效片元之后这样拟合效果更好
                loss+=torch.mean((hist-gt_hist).abs())
            else:
                loss+=torch.mean((hist[start_index:end_index]-gt_hist[start_index:end_index]).abs())
        
        loss=loss/sample_num
        loss.backward()
        loss_list.append(loss.item())

        if itr%50==0:
            save_ply(f"temp/result{itr}.ply",gaussians.means,gaussians.colours,gaussians.pre_act_opacities,gaussians.pre_act_scales,gaussians.pre_act_quats,colour_dim=1)
            # scipy.io.savemat(f"temp/hist{itr}.mat",{"hist":,"gt_hist":gt_hist.detach().cpu().numpy()})
            hist=hist.detach().cpu().numpy()
            gt_hist=gt_hist.detach().cpu().numpy()

            plt.figure(figsize=(8, 6))
            plt.plot(hist,label='hist')
            plt.plot(gt_hist,label='gt_hist')
            plt.legend(loc='upper right')
            plt.savefig(f"temp/hist{itr}.png")
            plt.close()

        with torch.no_grad():
        #     # 裁剪深度不在指定范围的片元
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
        #             # save_ply(f"temp/result{itr}_prune.ply",gaussians.means,gaussians.colours,gaussians.pre_act_opacities,gaussians.pre_act_scales,gaussians.pre_act_quats,colour_dim=1)

            # 一段时间要重置一次透明度，这样可以消除floaters悬浮物
            if itr==depth_prune_itr or itr % 500 == 0:
                prune_mask=torch.where(gaussians.get_colour<=1e-4, True, False).flatten()
                gaussians.prune_points(prune_mask)
                print(f"prune number: {torch.sum(prune_mask).item()}")

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            print(f"[*] Itr: {itr:07d} | Loss: {loss:0.3f} |")

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


# import os
# import torch
# import imageio
# import argparse
# import numpy as np

# from PIL import Image
# from tqdm import tqdm
# from torch.utils.data import DataLoader,WeightedRandomSampler
# from data_utils import save_ply,OptimizationParams,wasserstein_distance,TVLoss
# from pytorch3d.renderer.cameras import PerspectiveCameras,FoVPerspectiveCameras, look_at_view_transform

# from dataset import LCTDataset,ConfocalDataset
# import time
# import math
# import scipy
# import matplotlib.pyplot as plt

# ### 整个训练过程分为两步：
# ### 1.首先按照类似于BP的思路只按照深度来优化color
# ### 2.随后全面优化scale,opacity等参数

# from model2 import Scene, Gaussians

# def make_trainable(gaussians):

#     ### YOUR CODE HERE ###
#     # HINT: You can access and modify parameters from gaussians
#     gaussians.means.requires_grad=True
#     gaussians.pre_act_scales.requires_grad=True
#     gaussians.colours.requires_grad=True
#     gaussians.pre_act_opacities.requires_grad=True
    
#     if not gaussians.is_isotropic:
#         gaussians.pre_act_quats.requires_grad=True


# ### 随机初始化训练模型
# def run_training(args):
#     torch.manual_seed(16)

#     if not os.path.exists(args.out_path):
#         os.makedirs(args.out_path, exist_ok=True)

#     # # 随机初始化
#     # radius=0.65 ## cow数据的参数
#     # object_center=(0.0,0.0,1.3)
#     radius=0.25 ## mannequin数据的参数
#     object_center=(0.0,0.0,0.6)
#     gaussians = Gaussians(
#         num_points=5000, init_type="random",
#         device=args.device, isotropic=True,
#         colour_dim=1,extent=radius,center=object_center
#     )

#     fov_radius=1.2*radius

#     save_ply("temp/init.ply",gaussians.means,gaussians.colours,gaussians.pre_act_opacities,gaussians.pre_act_scales,gaussians.pre_act_quats,colour_dim=1)

#     scene = Scene(gaussians)
#     start=time.time()
    
#     start_index=150 # 参数是mannequin的
#     end_index=350
#     dataset= ConfocalDataset(args.data_path,device=args.device,is_train=True,start_index=start_index,end_index=end_index) 
#     img_size=(dataset.N,dataset.N) # 渲染图片大小
#     bin_resolution=dataset.bin_resolution
#     nums_bin=dataset.M

#     print("radius:",radius)
#     print("center:",object_center)
#     print("bin resolution:",bin_resolution)
#     print("width:",dataset.width)

#     sampler = WeightedRandomSampler(dataset.weights, len(dataset), replacement=True)
#     indices = torch.randperm(dataset.M)

#     train_loader = DataLoader(
#         dataset, batch_size=1,shuffle=True#, sampler=sampler
#     )
#     train_itr = iter(train_loader)
    
    