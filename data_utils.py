import os
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from plyfile import PlyData,PlyElement
from torch.utils.data import Dataset
from pytorch3d.renderer.cameras import PerspectiveCameras, look_at_view_transform

SH_C0 = 0.28209479177387814
CMAP_JET = plt.get_cmap("jet")
CMAP_MIN_NORM, CMAP_MAX_NORM = 0.5, 4.0

class CowDataset(Dataset):

    def __init__(self, root, split):
        super().__init__()
        self.root = root
        self.split = split
        if self.split not in ("train", "test"):
            raise ValueError(f"Invalid split: {self.split}")

        self.masks = []
        self.points = []
        self.images = []
        self.cameras = []

        imgs_root = os.path.join(root, "imgs")
        poses_root = os.path.join(root, "poses")
        points_root = os.path.join(root, "points")
        self.points_path = os.path.join(points_root, "points_10000.npy")

        data_img_size = None
        num_files = len(os.listdir(imgs_root))
        test_idxs = np.linspace(0, num_files, 7).astype(np.int32)[1:-1]
        test_idxs_set = set(test_idxs.tolist())
        train_idxs = [i for i in range(num_files) if i not in test_idxs_set]
        idxs = train_idxs if self.split == "train" else test_idxs

        for i in idxs:
            img_path = os.path.join(imgs_root, f"{i:03d}.png")
            npy_path = os.path.join(poses_root, f"{i:03d}.npy")

            img_ = imageio.v3.imread(img_path).astype(np.float32) / 255.0

            mask = None
            if img_.shape[-1] == 3:
                img = torch.tensor(img_)  # (H, W, 3)
            else:
                img = torch.tensor(img_[..., :3])  # (H, W, 3)
                mask = torch.tensor(img_[..., 3:4])  # (H, W, 1)

            img_size = img.shape[:2]
            dim = img_size[0]
            if img_size[0] != img_size[1]:
                raise RuntimeError

            # Checking if all data samples have the same image size
            if data_img_size is None:
                data_img_size = img_size
            else:
                if data_img_size[0] != img_size[0] or data_img_size[1] != img_size[1]:
                    raise RuntimeError

            pose = np.load(npy_path)
            dist, ele, az = pose.flatten()
            R, T = look_at_view_transform(dist, ele, az)

            img_size = img.shape[:2]
            camera = PerspectiveCameras(
                focal_length=5.0 * dim/2.0, in_ndc=False,
                principal_point=((dim/2, dim/2),),
                R=R, T=T, image_size=(img_size,),
            )

            self.images.append(img)
            self.cameras.append(camera)
            if mask is not None:
                self.masks.append(mask)

        self.img_size = data_img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        masks = None
        if len(self.masks) > 0:
            masks = self.masks[idx]
        return self.images[idx], self.cameras[idx], masks

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([x[0] for x in batch], dim=0)
        cameras = [x[1] for x in batch]

        masks = [x[2] for x in batch if x[2] is not None]
        if len(masks) == 0:
            masks = None
        else:
            masks = torch.stack(masks, dim=0)

        return images, cameras, masks

def colour_depth_q1_render(depth):
    normalized_depth = (depth - CMAP_MIN_NORM) / (CMAP_MAX_NORM - CMAP_MIN_NORM + 1e-8)
    coloured_depth = CMAP_JET(normalized_depth)[:, :, :3]  # (H, W, 3)
    coloured_depth = (np.clip(coloured_depth, 0.0, 1.0) * 255.0).astype(np.uint8)

    return coloured_depth

def visualize_renders(scene, gt_viz_img, cameras, img_size):

    imgs = []
    viz_size = (256, 256)
    with torch.no_grad():
        for cam in cameras:
            pred_img, _, _,_,_ = scene.render(
                cam, img_size=img_size,
                bg_colour=(0.0, 0.0, 0.0),
                per_splat=-1,
                no_grad=True
            )
            img = torch.clamp(pred_img, 0.0, 1.0) * 255.0
            img = img.clone().detach().cpu().numpy().astype(np.uint8)

            if img_size[0] != viz_size[0] or img_size[1] != viz_size[1]:
                img = np.array(Image.fromarray(img).resize(viz_size))

            imgs.append(img)

    pred_viz_img = np.concatenate(imgs, axis=1)
    viz_frame = np.concatenate((pred_viz_img, gt_viz_img), axis=0)
    return viz_frame

def construct_list_of_attributes(colours,scaling,rotation,features_rest):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(colours.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(features_rest.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

def save_ply(path,xyz_,colours_,opacity_,scaling_,rotation_,colour_dim=3):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError as error:
        print("Already has the directory!")
    
    max_sh_degree=3 # 球谐分量的部分
    f_rest_dim=3 * (max_sh_degree + 1) ** 2 - 3

    xyz = xyz_.detach().cpu().numpy() # [N,3]
    normals = np.zeros_like(xyz) # [N,3]
    if colour_dim==3:
        f_dc = colours_.detach().contiguous().cpu().numpy() # [N,3]
    else:
        f_dc = colours_.detach().contiguous().repeat(1,3).cpu().numpy()
    f_rest = np.zeros((f_dc.shape[0],f_rest_dim)) # 球谐分量, [N,f_rest_dim]
    opacities = opacity_.detach().cpu().unsqueeze(1).numpy() # [N,1]
    scale = scaling_.detach().cpu().numpy() # [N,1]
    rotation = rotation_.detach().cpu().numpy() # [N,4]
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(f_dc,scale,rotation,f_rest)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def load_gaussians_from_ply(path):
    # Modified from https://github.com/thomasantony/splat
    max_sh_degree = 3
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = xyz.astype(np.float32)
    rots = rots.astype(np.float32)
    scales = scales.astype(np.float32)
    opacities = opacities.astype(np.float32)
    shs = np.concatenate([
        features_dc.reshape(-1, 3),
        features_extra.reshape(len(features_dc), -1)
    ], axis=-1).astype(np.float32)
    shs = shs.astype(np.float32)

    dc_vals = shs[:, :3]
    dc_colours = np.maximum(dc_vals * SH_C0 + 0.5, np.zeros_like(dc_vals))

    output = {
        "xyz": xyz, "rot": rots, "scale": scales,
        "sh": shs, "opacity": opacities, "dc_colours": dc_colours
    }
    return output

def unproject_depth_image(depth, camera):
    """
    Unprojects a depth image into a 3D point cloud.

    Args:
        depth (torch.Tensor): The depth map of the image (S, S).
        camera: The Pytorch3D camera to render the image.
    
    Returns:
        points (torch.Tensor): The 3D points of the unprojected image (N, 3).
        rgba (torch.Tensor): The rgba color values corresponding to the unprojected
            points (N, 4).
    """
    device = camera.device
    ndc_pixel_coordinates = torch.linspace(1, -1, depth.shape[0])
    Y, X = torch.meshgrid(ndc_pixel_coordinates, ndc_pixel_coordinates)
    xy_depth = torch.dstack([X, Y, depth])
    points = camera.unproject_points(
        xy_depth.to(device), in_ndc=False, from_ndc=False, world_coordinates=True,
    )

    return points

def colours_from_spherical_harmonics(spherical_harmonics, gaussian_dirs):
    """
    [Q 1.3.1] Computes view-dependent colour given spherical harmonic coefficients
    and direction vectors for each gaussian.

    Args:
        spherical_harmonics     :   A torch.Tensor of shape (N, 48) representing the
                                    spherical harmonic coefficients.
        gaussian_dirs           :   A torch.Tensor of shape (N, 3) representing the
                                    direction vectors pointing from the camera origin
                                    to each Gaussian.

    Returns:
        colours                 :   A torch.Tensor of shape (N, 3) representing the view dependent
                                    RGB colour.
    """
    ### YOUR CODE HERE ###
    SH_C0 = 0.28209479177387814
    SH_C1 = 0.4886025119029199
    SH_C2_0 = 1.0925484305920792
    SH_C2_1 = -1.0925484305920792
    SH_C2_2 = 0.31539156525252005
    SH_C2_3 = -1.0925484305920792
    SH_C2_4 = 0.5462742152960396
    SH_C3_0 = -0.5900435899266435
    SH_C3_1 = 2.890611442640554
    SH_C3_2 = -0.4570457994644658
    SH_C3_3 = 0.3731763325901154
    SH_C3_4 = -0.4570457994644658
    SH_C3_5 = 1.445305721320277
    SH_C3_6 = -0.5900435899266435

    # 0阶球谐函数
    color = spherical_harmonics[:,0:3]*SH_C0

    # 1阶球谐函数
    x = gaussian_dirs[:,0:1].repeat((1,3))
    y = gaussian_dirs[:,1:2].repeat((1,3))
    z = gaussian_dirs[:,2:3].repeat((1,3))
    color = color - SH_C1 * y * spherical_harmonics[:,3:6] + SH_C1 * z * spherical_harmonics[:,6:9] - SH_C1 * x * spherical_harmonics[:,9:12]

    # 2阶球谐函数
    (xx, yy, zz) = (x * x, y * y, z * z)
    (xy, yz, xz) = (x * y, y * z, x * z)
    color = color +	SH_C2_0 * xy * spherical_harmonics[:,12:15] + \
        SH_C2_1 * yz * spherical_harmonics[:,15:18] + \
        SH_C2_2 * (2.0 * zz - xx - yy) * spherical_harmonics[:,18:21] + \
        SH_C2_3 * xz * spherical_harmonics[:,21:24] + \
        SH_C2_4 * (xx - yy) * spherical_harmonics[:,24:27]
    
    # 3阶球谐函数
    color = color + \
                SH_C3_0 * y * (3.0 * xx - yy) * spherical_harmonics[:,27:30] + \
                SH_C3_1 * xy * z * spherical_harmonics[:,30:33] + \
                SH_C3_2 * y * (4.0 * zz - xx - yy) * spherical_harmonics[:,33:36] + \
                SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) *spherical_harmonics[:,36:39] + \
                SH_C3_4 * x * (4.0 * zz - xx - yy) * spherical_harmonics[:,39:42] + \
                SH_C3_5 * z * (xx - yy) * spherical_harmonics[:,42:45] + \
                SH_C3_6 * x * (xx - 3.0 * yy) * spherical_harmonics[:,45:48]
        
    color += 0.5
    return torch.clip(color, 0.0, 1.0)
