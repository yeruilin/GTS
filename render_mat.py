# 以多视角的方式渲染voxel

import os
import torch
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from scene import Scene
from gaussian import Gaussians
from data_utils import colour_depth_q1_render,save_ply
from pytorch3d.renderer.cameras import PerspectiveCameras, look_at_view_transform,FoVPerspectiveCameras

from scipy.ndimage import uniform_filter,convolve,gaussian_filter

import cv2
import scipy
import math

def create_renders(args):

    dim = args.img_dim
    img_size = (dim, dim)

    frame_rate=5.0
    num_views = 32
    azims = np.linspace(-180, 180, num_views)

    debug_root = os.path.join(args.out_path, "render")
    if not os.path.exists(debug_root):
        os.makedirs(debug_root, exist_ok=True)

    # load voxel
    mat_path=args.data_path # bunny_result
    thresh=0.1

    mat_data = scipy.io.loadmat(mat_path)
    rho = mat_data['rho']# .transpose(1,0,2)
    rho=rho/np.max(rho)

    if "opacity" in mat_data.keys():
        opacity=np.squeeze(mat_data['opacity'])#.transpose(1,0,2)
    else:
        opacity=np.zeros(rho.shape,dtype=np.float32)
    
    # min_pos=[-0.3,-0.3,-0.15] ## frontback_christ数据参数
    # max_pos=[0.3,0.3,0.15]

    # min_pos=[-0.35,-0.35,-0.3] ## frontback_bunny数据参数
    # max_pos=[0.35,0.35,0.3]

    # min_pos=[-0.32,-0.3,-0.3] ## frontback_david数据参数
    # max_pos=[0.32,0.3,0.3]

    # min_pos=[-0.2,-0.3,-0.2] ## frontback_hydrant数据参数
    # max_pos=[0.2,0.3,0.2]

    # min_pos=[-0.35,-0.35,-0.3] ## frontback_bunny_exp数据参数
    # max_pos=[0.35,0.35,0.3]

    # min_pos=[-0.3,-0.5,-0.3] ## frontback_lion_exp数据参数
    # max_pos=[0.5,0.5,0.3]

    # min_pos=[-0.5,-0.5,-0.2] ## front_lion的参数
    # max_pos=[0.5,0.5,0.2]
    
    min_pos=[-0.3,-0.3,-0.3]
    max_pos=[0.3,0.3,0.3]

    lower_corner = np.array(min_pos, dtype=float)
    upper_corner = np.array(max_pos, dtype=float)
    
    grid_shape = np.array(rho.shape, dtype=float)
    steps = (upper_corner - lower_corner) / (grid_shape - 1) # 计算每个维度的步长

    if "scale" in mat_data.keys():
        # scale=np.log(mat_data['scale'])
        scale=np.log(np.ones(rho.shape,dtype=np.float32)*steps[0])
    else:
        scale=np.log(np.ones(rho.shape,dtype=np.float32)*steps[0])
    
    # 获取大于阈值的坐标和值
    x_idx, y_idx, z_idx = np.where(rho > thresh)
    rho = rho[x_idx, y_idx, z_idx]
    opacity=opacity[x_idx, y_idx, z_idx]
    scale=scale[x_idx, y_idx, z_idx]
    
    # 构建4维向量 [x,y,z,value]
    result = np.column_stack((x_idx, y_idx, z_idx,rho,opacity,scale))
    result = result.reshape(-1, 6)
    print(result.shape)

    xyz=lower_corner + result[:,:3]*steps # 均值
    rho=result[:,3:4] # 反射率
    opacity=result[:,4:5] # 透明度
    scale=result[:,5:6] # 大小

    # 计算xyz中心点
    gaussians = Gaussians(
        num_points=100, init_type="random",
        device=args.device, isotropic=True,
        colour_dim=1,scale=steps[0]
    )

    filename=args.data_path.split("/")[-1][:-4]

    gaussians.means = torch.from_numpy(xyz).to(args.device).float()
    gaussians.colours=torch.from_numpy(rho).to(args.device).float()
    gaussians.opacities=torch.from_numpy(opacity).to(args.device).float()
    gaussians.scales=torch.from_numpy(scale).to(args.device).float()

    _range=torch.max(gaussians.means,dim=0)[0]-torch.min(gaussians.means,dim=0)[0]
    gaussians.radius=torch.max(_range/2.0).to(args.device)

    scene = Scene(gaussians)

    bg_colour=(1,1,1) # 背景颜色
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
            img, depth, mask= scene.render(camera,args.gaussians_per_splat,img_size,no_grad=True)

        img=(img-torch.min(img))/(torch.max(img)-torch.min(img))
        img[(mask<0.5).expand(-1, -1, 3)]=1.0

        debug_path = os.path.join(debug_root, f"{i:03d}.png")
        img = img.detach().cpu().numpy()
        mask = mask.repeat(1, 1, 3).detach().cpu().numpy()
        depth = depth.detach().cpu().numpy()

        if use_filter:
            img = cv2.medianBlur(img, 5)
            img = gaussian_filter(img, sigma=1.0)

        img = (img * 255.0).astype(np.uint8)
        mask = np.where(mask > 0.5, 255.0, 0.0).astype(np.uint8)  # (H, W, 3)

        # Colouring the depth map
        depth = depth[:, :, 0].astype(np.float32)  # (H, W) # 有效的depth在5-7之间，因此可以在这个范围归一化配置颜色
        if use_filter:
            depth = cv2.medianBlur(depth, 5)
            depth = gaussian_filter(depth, sigma=1.0)
        coloured_depth = colour_depth_q1_render(depth)  # (H, W, 3)

        # ## 旋转180°
        img=np.flipud(np.fliplr(img))
        coloured_depth=np.flipud(np.fliplr(coloured_depth))
        mask=np.flipud(np.fliplr(mask))

        # # ## 左右翻转(phasor数据都需要左右翻转)
        # img=np.fliplr(img)
        # coloured_depth=np.fliplr(coloured_depth)
        # mask=np.fliplr(mask)

        # ## 旋转90°
        # img=np.rot90(img)
        # coloured_depth=np.rot90(coloured_depth)
        # mask=np.rot90(mask)

        concat = np.concatenate([img,coloured_depth, mask], axis = 1)
        resized = Image.fromarray(concat).resize((256*3, 256))
        resized.save(debug_path)

        imgs.append(np.array(resized))

    gif_path = os.path.join(args.out_path, f"{filename}.gif")
    imageio.mimwrite(gif_path, imgs, duration=1000.0*(1/frame_rate), loop=0)
    

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path", default="./output", type=str,
        help="Path to the directory where output should be saved to."
    )
    parser.add_argument(
        "--data_path", default="./temp/result500.mat", type=str,
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
