import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from model import Scene, Gaussians
from torch.utils.data import DataLoader
from data_utils import visualize_renders
from data_utils_harder_scene import get_nerf_datasets, trivial_collate

from pytorch3d.renderer import PerspectiveCameras
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from SSIM import ssim

class OptimizationParams:
    def __init__(self):
        self.iterations = 1100
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100 # 重置透明度之后，间隔100次重新增删片元密度
        self.opacity_reset_interval = 400 # 重置透明度的间隔
        self.densify_from_iter = 400 # 超过这个阈值会重置一次透明度
        self.densify_until_iter = 15_000 # 前15000次都需要增加高斯密度
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"

def make_trainable(gaussians):

    ### YOUR CODE HERE ###
    # HINT: You can access and modify parameters from gaussians
    gaussians.means.requires_grad=True
    gaussians.pre_act_scales.requires_grad=True
    gaussians.colours.requires_grad=True
    gaussians.pre_act_opacities.requires_grad=True
    
    if not gaussians.is_isotropic:
        gaussians.pre_act_quats.requires_grad=True

def ndc_to_screen_camera(camera, img_size = (128, 128)):

    min_size = min(img_size[0], img_size[1])

    screen_focal = camera.focal_length * min_size / 2.0
    screen_principal = torch.tensor([[img_size[0]/2, img_size[1]/2]]).to(torch.float32)

    return PerspectiveCameras(
        R=camera.R, T=camera.T, in_ndc=False,
        focal_length=screen_focal, principal_point=screen_principal,
        image_size=(img_size,),
    )

def run_training(args):

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    train_dataset, val_dataset, _ = get_nerf_datasets(
        dataset_name="materials", data_root=args.data_path,
        image_size=[128, 128],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    train_itr = iter(train_loader)

    # Preparing some code for visualization
    viz_gif_path_1 = os.path.join(args.out_path, "q1_harder_training_progress.gif")
    viz_gif_path_2 = os.path.join(args.out_path, "q1_harder_training_final_renders.gif")
    viz_idxs = np.linspace(0, len(train_dataset)-1, 5).astype(np.int32)[:4]

    gt_viz_imgs = [(train_dataset[i]["image"]*255.0).numpy().astype(np.uint8) for i in viz_idxs]
    gt_viz_imgs = [np.array(Image.fromarray(x).resize((256, 256))) for x in gt_viz_imgs]
    gt_viz_img = np.concatenate(gt_viz_imgs, axis=1)

    viz_cameras = [ndc_to_screen_camera(train_dataset[i]["camera"]).to(args.device) for i in viz_idxs]

    # Init gaussians and scene
    gaussians = Gaussians(
        num_points=10000, init_type="random",
        device=args.device, isotropic=False
    )
    scene = Scene(gaussians)

    # Making gaussians trainable and setting up optimizer
    make_trainable(gaussians)
    opt_param=OptimizationParams() # 设置优化参数
    gaussians.training_setup(opt_param) # 设置优化模式

    bg_colour=(0.0,0.0,0.0) # 白色背景

    # Training loop
    viz_frames = []
    for itr in range(1,opt_param.iterations):

        # Fetching data
        try:
            data = next(train_itr)
        except StopIteration:
            train_itr = iter(train_loader)
            data = next(train_itr)

        gt_img = data[0]["image"].to(args.device)
        camera = ndc_to_screen_camera(data[0]["camera"]).to(args.device)

        gaussians.update_learning_rate(itr) # 更新学习率

        # Rendering scene using gaussian splatting
        pred_img, depth, mask,means_2D,radii = scene.render(camera,args.gaussians_per_splat,(gt_img.shape[0],gt_img.shape[1]),bg_colour)

        visibility_filter= (radii > 0).nonzero() # 选出所有在screen上大小超过0的索引

        # Compute loss
        ### YOUR CODE HERE ###
        L1loss = torch.mean((pred_img-gt_img).abs()) # L1 Loss
        SSIMloss=1-ssim(pred_img.permute(2,1,0).unsqueeze(0),gt_img.permute(2,1,0).unsqueeze(0)) ## SSIM Loss
        loss=(1-opt_param.lambda_dssim)*L1loss+opt_param.lambda_dssim*SSIMloss

        loss.backward()

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
                    size_threshold = 20 if itr > opt_param.opacity_reset_interval else None # 片元在图片上的大小不超过20
                    gaussians.densify_and_prune(
                        grad_threshold=opt_param.densify_grad_threshold, 
                        min_opacity=0.005, 
                        extent=scene.cameras_extent, 
                        max_screen_size=size_threshold, 
                        radii=radii
                    )
                
                # 一段时间要重置一次透明度，这样可以消除floaters悬浮物
                if itr % opt_param.opacity_reset_interval == 0 or (itr == opt_param.densify_from_iter):
                    print("reset_opacity")
                    gaussians.reset_opacity()

            # step的过程会为每个变量创建一个字典，包括step，exp_avg等优化参数，所以我们导入优化参数都用load_state_dict
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)

        print(f"[*] Itr: {itr:07d} | Loss: {loss:0.3f}")

        if itr % args.viz_freq == 0:
            viz_frame = visualize_renders(
                scene, gt_viz_img,
                viz_cameras, (128, 128)
            )
            viz_frames.append(viz_frame)

    print("[*] Training Completed.")

    # Saving training progess GIF
    imageio.mimwrite(viz_gif_path_1, viz_frames, loop=0, duration=(1/10.0)*1000)

    # Creating renderings of the training views after training is completed.
    frames = []
    viz_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    for viz_data in tqdm(viz_loader, desc="Creating Visualization"):
        gt_img = viz_data[0]["image"].to(args.device)
        camera = ndc_to_screen_camera(viz_data[0]["camera"]).to(args.device)

        with torch.no_grad():

            # Rendering scene using gaussian splatting
            pred_img, depth, mask,means_2D,radii = scene.render(camera,args.gaussians_per_splat,(gt_img.shape[0],gt_img.shape[1]),bg_colour,no_grad=True)

        pred_npy = pred_img.detach().cpu().numpy()
        pred_npy = (np.clip(pred_npy, 0.0, 1.0) * 255.0).astype(np.uint8)
        frames.append(pred_npy)

    # Saving renderings
    imageio.mimwrite(viz_gif_path_2, frames, loop=0, duration=(1/10.0)*1000)

    # Running evaluation using the test dataset
    psnr_vals, ssim_vals = [], []
    for val_data in tqdm(val_loader, desc="Running Evaluation"):

        gt_img = val_data[0]["image"].to(args.device)
        camera = ndc_to_screen_camera(val_data[0]["camera"]).to(args.device)

        with torch.no_grad():

            # Rendering scene using gaussian splatting
            pred_img, depth, mask,means_2D,radii = scene.render(camera,args.gaussians_per_splat,(gt_img.shape[0],gt_img.shape[1]),bg_colour,no_grad=True)

            gt_npy = gt_img.detach().cpu().numpy()
            pred_npy = pred_img.detach().cpu().numpy()
            psnr = peak_signal_noise_ratio(gt_npy, pred_npy)
            ssim_val = structural_similarity(gt_npy, pred_npy, channel_axis=-1, data_range=1.0)

            psnr_vals.append(psnr)
            ssim_vals.append(ssim_val)

    mean_psnr = np.mean(psnr_vals)
    mean_ssim = np.mean(ssim_vals)
    print(f"[*] Evaluation --- Mean PSNR: {mean_psnr:.3f}")
    print(f"[*] Evaluation --- Mean SSIM: {mean_ssim:.3f}")

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path", default="./output", type=str,
        help="Path to the directory where output should be saved to."
    )
    parser.add_argument(
        "--data_path", default="./data/materials", type=str,
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
