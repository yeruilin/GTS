import os
import torch
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from model_old import Gaussians, Scene
from data_utils import colour_depth_q1_render
from pytorch3d.renderer.cameras import PerspectiveCameras,FoVPerspectiveCameras, look_at_view_transform

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

    # Preprocessing for ease of rendering
    new_points = gaussians.means - gaussians.means.mean(dim=0, keepdims=True)
    gaussians.means = new_points

    print(torch.min(gaussians.colours))
    print(torch.max(gaussians.colours))
    print(torch.mean(gaussians.colours))
    print(torch.median(gaussians.colours))

    mask=(gaussians.colours[:,0]>0.523).squeeze()

    gaussians.colours=gaussians.colours[mask]
    gaussians.pre_act_opacities=gaussians.pre_act_opacities[mask]
    gaussians.pre_act_quats=gaussians.pre_act_quats[mask]
    gaussians.pre_act_scales=gaussians.pre_act_scales[mask]
    gaussians.means=gaussians.means[mask]

    _range=torch.max(gaussians.means,dim=0)[0]-torch.min(gaussians.means,dim=0)[0]
    radius=0.5*torch.max(_range)
    print("radius:",radius)

    # Creating the scene with the loaded gaussians
    scene = Scene(gaussians)

    bg_colour=(1.0, 1.0, 1.0) # 背景颜色

    imgs = []
    for i in tqdm(range(num_views), desc="Rendering"):
        dist = gaussians.radius.item()*8
        R, T = look_at_view_transform(dist = dist, azim=azims[i], elev=30.0, up=((0, 1, 0),))
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

        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        mask = np.where(mask > 0.5, 255.0, 0.0).astype(np.uint8)  # (H, W, 3)

        # Colouring the depth map
        depth = depth[:, :, 0].astype(np.float32)  # (H, W) # 有效的depth在5-7之间，因此可以在这个范围归一化配置颜色
        coloured_depth = colour_depth_q1_render(depth)  # (H, W, 3)

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
