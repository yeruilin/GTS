import os
import torch
import argparse
import numpy as np
import math
from PIL import Image
from data_utils import colour_depth_q1_render
from model import Gaussians, Scene
from pytorch3d.renderer.cameras import PerspectiveCameras,FoVPerspectiveCameras, look_at_view_transform

from scipy.io import savemat

def create_renders(args):

    dim = args.img_dim
    img_size = (dim, dim)

    debug_root = os.path.join(args.out_path, "q1_render")
    if not os.path.exists(debug_root):
        os.makedirs(debug_root, exist_ok=True)

    # Loading pre-trained gaussians
    gaussians = Gaussians(
        load_path=args.data_path, init_type="gaussians",
        device=args.device
    )

    # Preprocessing for ease of rendering：将高斯场景中心移动到原点
    new_points = gaussians.means - gaussians.means.mean(dim=0, keepdims=True)
    gaussians.means = new_points
    radius=torch.sqrt(new_points[:,0]**2+new_points[:,1]**2+new_points[:,2]**2)
    radius=torch.max(radius) # 球体半径
    radius=radius.item() # float
    print("radius",radius)

    # Creating the scene with the loaded gaussians
    scene = Scene(gaussians)

    bg_colour=(1.0, 1.0, 1.0) # 背景颜色
    step=args.width/(args.img_dim-1) # 横向和纵向的扫描范围
    object_center=(0,0,0) # 场景中心

    transient_map=np.zeros((args.bin,args.img_dim,args.img_dim),dtype=np.float32)

    with torch.no_grad():
        for i in range(img_size[0]):
            for j in range(img_size[1]):
                scan_point=(-args.width/2+step*i,-args.width/2+step*j,-2.0) # 扫描点在z=2.0的位置
                dist=math.sqrt((scan_point[0]-object_center[0])**2+(scan_point[1]-object_center[1])**2+(scan_point[2]-object_center[2])**2) # 扫描点到场景中心的距离
                fov=math.asin(radius/dist)
                R, T = look_at_view_transform(eye=(scan_point,),at=(object_center,),up=((0, 1, 0),)) # 因为高斯元中心在原点，因此at就是原点
                current_camera = FoVPerspectiveCameras(
                    znear=0.1,zfar=10.0,
                    fov=fov,degrees=False, # radian
                    R=R, T=T
                ).to(args.device)
                current_camera.image_size=(img_size,)

                img, depth, mask,_,_ = scene.render(current_camera,args.gaussians_per_splat,img_size,bg_colour,no_grad=True)

                debug_path = os.path.join(debug_root, f"{i:03d}.png")
                img = img.detach().cpu().numpy()
                mask = mask.repeat(1, 1, 3).detach().cpu().numpy()
                depth = depth.detach().cpu().numpy()

                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
                mask = np.where(mask > 0.5, 255.0, 0.0).astype(np.uint8)  # (H, W, 3)

                # Colouring the depth map
                depth = depth[:, :, 0].astype(np.float32)  # (H, W) # 有效的depth在5-7之间，因此可以在这个范围归一化配置颜色
                coloured_depth = colour_depth_q1_render(depth)  # (H, W, 3)

                concat = np.concatenate([img, coloured_depth, mask], axis = 1)
                resized = Image.fromarray(concat).resize((256*3, 256))
                resized.save(debug_path)

                exit()

                # # Rendering histogram using gaussian splatting
                # hist = scene.render_conf_hist(current_camera,args.gaussians_per_splat,img_size,bg_colour,no_grad=True)
                # transient_map[:,i,j]=hist.detach().cpu().numpy()

        # 将瞬态图保存为.mat文件
        savemat("confocal_cow.mat",{"data":transient_map})

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
        "--img_dim", default=64, type=int,
        help=(
            "Spatial dimension of the rendered image. "
            "The rendered image will have img_dim as its height and width."
        )
    )
    parser.add_argument(
        "--bin", default=512, type=int,
        help=(
            "Histogram bin in transient map. "
        )
    )
    parser.add_argument(
        "--bin_resolution", default=0.02, type=int,
        help=(
            "Bin resolution. "
        )
    )
    parser.add_argument(
        "--width", default=2.5, type=int,
        help=(
            "Scanning width"
        )
    )
    parser.add_argument(
        "--gaussians_per_splat", default=2048, type=int,
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
