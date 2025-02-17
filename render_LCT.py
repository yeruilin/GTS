import os
import torch
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from model_old import Gaussians, Scene
# from model2 import Gaussians,Scene
from data_utils import colour_depth_q1_render,save_ply
from pytorch3d.renderer.cameras import PerspectiveCameras,FoVPerspectiveCameras, look_at_view_transform

import math
import matplotlib.pyplot as plt

from scipy.ndimage import uniform_filter,convolve,gaussian_filter

import cv2

def create_renders(args):

    dim = args.img_dim
    img_size = (dim, dim)

    num_views = 32
    azims = np.linspace(-180, 180, num_views)

    debug_root = os.path.join(args.out_path, "q1_render")
    if not os.path.exists(debug_root):
        os.makedirs(debug_root, exist_ok=True)

    # Loading pre-trained gaussians
    gaussians = Gaussians(
        load_path=args.data_path, init_type="gaussians",
        device=args.device
    )

    filename=args.data_path.split("/")[-1][:-4]

    print("colour max:",torch.max(gaussians.get_colour))
    print("colour mean:",torch.mean(gaussians.get_colour))
    print("colour median:",torch.median(gaussians.get_colour))

    print(torch.max(gaussians.get_scaling))

    mask=(gaussians.get_colour[:,0]>torch.max(torch.mean(gaussians.get_colour),torch.median(gaussians.get_colour))).squeeze()
    mask=(gaussians.get_colour[:,0]>0.0035).squeeze()
    ## mask=(gaussians.means[:,2]<1.25).squeeze()

    gaussians.colours=gaussians.colours[mask]
    gaussians.pre_act_opacities=gaussians.pre_act_opacities[mask]
    gaussians.pre_act_quats=gaussians.pre_act_quats[mask]
    gaussians.pre_act_scales=gaussians.pre_act_scales[mask]
    gaussians.means=gaussians.means[mask]

    print(torch.max(gaussians.means,dim=0)[0])
    print(torch.min(gaussians.means,dim=0)[0])

    object_center=gaussians.means.mean(dim=0, keepdims=True).flatten().detach().cpu().numpy()
    object_center=(object_center[0],object_center[1],object_center[2])
    print(object_center)

    _range=torch.max(gaussians.means,dim=0)[0]-torch.min(gaussians.means,dim=0)[0]
    radius=0.5*torch.max(_range)
    print("radius:",radius)

    print(torch.max(gaussians.get_scaling))

    # Creating the scene with the loaded gaussians
    scene = Scene(gaussians)

    # Preprocessing for ease of rendering
    new_points = gaussians.means - gaussians.means.mean(dim=0, keepdims=True)
    gaussians.means = new_points

    scene = Scene(gaussians)

    bg_colour=(0.0, 0.0, 0.0) # 背景颜色
    use_filter=False # 后处理

    imgs = []
    for i in tqdm(range(num_views), desc="Rendering"):
        dist = gaussians.radius.item()*8
        R, T = look_at_view_transform(dist = dist, azim=azims[i], elev=0.0, up=((0, 1, 0),))
        camera = PerspectiveCameras(
            focal_length=5.0 * dim/2.0, in_ndc=False,
            principal_point=((dim/2, dim/2),),
            R=R, T=T, image_size=(img_size,),
        ).to(args.device)

        with torch.no_grad():
            # Rendering scene using gaussian splatting
            img, depth, mask,_,_ = scene.render(camera,args.gaussians_per_splat,img_size,bg_colour,no_grad=True)

        debug_path = os.path.join(debug_root, f"{i:03d}.png")
        img = img.detach().cpu().numpy()
        mask = mask.repeat(1, 1, 3).detach().cpu().numpy()
        depth = depth.detach().cpu().numpy()

        if use_filter:
            img = cv2.medianBlur(img, 3)
            img = gaussian_filter(img, sigma=1.0, truncate=2.5)

        img=(img-np.min(img))/(np.max(img)-np.min(img))
        img = (img * 255.0).astype(np.uint8)
        mask = np.where(mask > 0.5, 255.0, 0.0).astype(np.uint8)  # (H, W, 3)

        # Colouring the depth map
        depth = depth[:, :, 0].astype(np.float32)  # (H, W) # 有效的depth在5-7之间，因此可以在这个范围归一化配置颜色
        if use_filter:
            depth = cv2.medianBlur(depth, 3)
            depth = gaussian_filter(depth, sigma=1.0, truncate=2.5)
        coloured_depth = colour_depth_q1_render(depth)  # (H, W, 3)

        ## 旋转180°
        img=np.flipud(np.fliplr(img))
        coloured_depth=np.flipud(np.fliplr(coloured_depth))
        mask=np.flipud(np.fliplr(mask))

        # ## 旋转90°
        # img=np.rot90(img)
        # coloured_depth=np.rot90(coloured_depth)
        # mask=np.rot90(mask)

        concat = np.concatenate([img, coloured_depth, mask], axis = 1)
        resized = Image.fromarray(concat).resize((256*3, 256))
        resized.save(debug_path)

        imgs.append(np.array(resized))

    gif_path = os.path.join(args.out_path, f"{filename}.gif")
    imageio.mimwrite(gif_path, imgs, duration=1000.0*(1/10.0), loop=0)
    

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path", default="./output", type=str,
        help="Path to the directory where output should be saved to."
    )
    parser.add_argument(
        "--data_path", default="./data/sledge.ply", type=str,
        help="Path to the pre-trained gaussian data to be rendered."
    )
    parser.add_argument(
        "--img_dim", default=256, type=int,
        help=(
            "Spatial dimension of the rendered image. "
            "The rendered image will have img_dim as its height and width."
        )
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
    parser.add_argument("--device", default="cuda:1", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    create_renders(args)
