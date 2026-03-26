import os
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from plyfile import PlyData,PlyElement
from torch.utils.data import Dataset
from pytorch3d.renderer.cameras import PerspectiveCameras,FoVPerspectiveCameras, look_at_view_transform
import math
import scipy

SH_C0 = 0.28209479177387814
CMAP_JET = plt.get_cmap("jet")
CMAP_MIN_NORM, CMAP_MAX_NORM = 1.8,2.5

class OptimizationParams:
    def __init__(self):
        self.iterations = 30000
        self.position_lr_init = 0
        self.position_lr_final = 0
        self.position_lr_delay_mult = 0.001
        self.position_lr_max_steps = 30000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 1000 # 重置透明度之后，间隔100次重新增删片元密度
        self.opacity_reset_interval = 50 # 重置透明度的间隔
        self.densify_from_iter = 5000 # 超过这个阈值会重置一次透明度
        self.densify_until_iter = 15000 # 前15000次都需要增加高斯密度
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"

def plot_hist(hist,gt_hist,itr):
    hist=hist.detach().cpu().numpy()
    gt_hist=gt_hist.detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.plot(hist,label='hist')
    plt.plot(gt_hist,label='gt_hist')
    plt.legend(loc='upper right')
    plt.savefig(f"temp/hist{itr}.png")
    plt.close()

# 根据扫描点的位置和场景中心，构建一个FoV相机
def get_camera(scan_point,object_center,fov_radius,img_size,device):
    dist=math.sqrt((scan_point[0]-object_center[0])**2+(scan_point[1]-object_center[1])**2+(scan_point[2]-object_center[2])**2) # 扫描点到场景中心的距离
    fov=2*math.asin(fov_radius/dist)
    R, T = look_at_view_transform(eye=(scan_point,),at=(object_center,),up=((0, 1, 0),)) # 因为高斯元中心在原点，因此at就是原点
    current_camera = FoVPerspectiveCameras(
        znear=0.1,zfar=10.0,
        fov=fov,degrees=False, # radian
        R=R, T=T
    )
    current_camera.image_size=(img_size,)

    return current_camera.to(device)
    
def TVLoss(image):
    """
    计算图像的总变差损失。
    
    参数:
    image (torch.Tensor): 形状为[H, W]的图像张量。
    
    返回:
    torch.Tensor: 图像的TV损失。
    """
    # 计算水平方向的TV损失
    tv_h = torch.mean(torch.abs(image[:, 1:] - image[:, :-1]))
    
    # 计算垂直方向的TV损失
    tv_w = torch.mean(torch.abs(image[1:, :] - image[:-1, :]))
    
    # 返回总TV损失
    return (tv_h + tv_w)/2


def wasserstein_distance(p, gt,indices=None):
    """
    计算一维离散分布 p 和 q 的 Wasserstein 距离
    Args:
        p (torch.Tensor): 第一个分布，形状为 [N]
        gt (torch.Tensor): 第二个分布，形状为 [N]
    Returns:
        float: Wasserstein 距离
    """
    p=p.flatten()
    gt=gt.flatten()

    if indices!=None:
        p = p[indices]
        gt=gt[indices]

    # 计算 Wasserstein 距离
    cdf_p = torch.cumsum(p, dim=0)  # 累积分布函数 (CDF)
    cdf_q = torch.cumsum(gt, dim=0)

    wasserstein_dist1 = torch.mean(torch.abs(cdf_p - cdf_q))#/torch.sum(gt)

    p = torch.flip(p, dims=[0])
    gt = torch.flip(gt, dims=[0])

    cdf_p2 = torch.cumsum(p, dim=0)  # 累积分布函数 (CDF)
    cdf_q2 = torch.cumsum(gt, dim=0)
    wasserstein_dist2 = torch.mean(torch.abs(cdf_p2 - cdf_q2))

    wasserstein_dist=(wasserstein_dist1+wasserstein_dist2)/2

    return wasserstein_dist

def wasserstein_distance2(p, gt):
    """
    计算一维离散分布 p 和 q 的 Wasserstein 距离
    Args:
        p (torch.Tensor): 第一个分布，形状为 [N]
        gt (torch.Tensor): 第二个分布，形状为 [N]
    Returns:
        float: Wasserstein 距离
    """
    p=p.flatten()/(p.sum()+1e-5)
    gt=gt.flatten()/gt.sum()

    # 计算 Wasserstein 距离
    cdf_p = torch.cumsum(p, dim=0)  # 累积分布函数 (CDF)
    cdf_q = torch.cumsum(gt, dim=0)

    wasserstein_dist1 = torch.mean(torch.abs(cdf_p - cdf_q))#/torch.sum(gt)

    p = torch.flip(p, dims=[0])
    gt = torch.flip(gt, dims=[0])

    # 计算 Wasserstein 距离
    cdf_p2 = torch.cumsum(p, dim=0)  # 累积分布函数 (CDF)
    cdf_q2 = torch.cumsum(gt, dim=0)

    wasserstein_dist2 = torch.mean(torch.abs(cdf_p2 - cdf_q2))#/torch.sum(gt)

    wasserstein_dist=(wasserstein_dist1+wasserstein_dist2)/2

    return wasserstein_dist

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


def makegrid(minimalpos, maximalpos, gridsize):
    minimalpos = np.asarray(minimalpos)
    maximalpos = np.asarray(maximalpos)
    gridsize = np.asarray(gridsize)

    # Number of pixels per direction
    pixels = np.ceil(np.abs(maximalpos - minimalpos)/gridsize).astype(np.int32)
    gridsize=np.abs(maximalpos - minimalpos)/pixels # 重新计算步长

    # Unit vectors scaled by grid size
    vx = np.array([1, 0, 0]) * gridsize
    vy = np.array([0, 1, 0]) * gridsize
    vz = np.array([0, 0, 1]) * gridsize

    # Generate grid points
    pts = []
    for x in range(pixels[0]):
        for y in range(pixels[1]):
            for z in range(pixels[2]):
                tmp = minimalpos + (x+0.5) * vx + (y+0.5) * vy + z * vz
                pts.append(tmp)

    pts = np.array(pts)
    return pts,pixels,gridsize


def scatter_trilinear(means, vals, min_pos, grid_size, pixels):
    """
    将 means(N,d) 的数值 vals(N) 按三线性权重分散到 3D 体素网格。

    参数
    ----
    means     : (N,3) 世界坐标
    vals      : (N,)  或 (N,k) 要写入的标量/向量
    min_pos   : (3,)  网格原点
    grid_size : (3,)  体素物理尺寸
    pixels    : (3,)  网格分辨率 [nx,ny,nz]
    
    返回
    ----
    out       : 累加后的网格
    """
    
    nx, ny, nz = pixels

    # 体素坐标 [0,nx-1] 等
    xyz = (means - min_pos) / grid_size        # (N,3)
    xyz = np.clip(xyz, 0, [nx-1, ny-1, nz-1])
    idx0 = xyz.astype(int)                     # 左下角 voxel 索引
    frac = xyz - idx0                          # 局部小数部分 [0,1)

    # 8 个邻居偏移
    offset = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                       [1,1,0],[1,0,1],[0,1,1],[1,1,1]])
    w = np.empty((8, means.shape[0]))
    for i in range(8):
        dx, dy, dz = offset[i]
        w[i] = (1-dx+(-1)**dx*frac[:,0]) * \
               (1-dy+(-1)**dy*frac[:,1]) * \
               (1-dz+(-1)**dz*frac[:,2])
    w = np.clip(w, 0, 1)
    w /= (w.sum(0, keepdims=True) + 1e-6)       # 归一化
    
    out = np.zeros((nx, ny, nz), dtype=vals.dtype)
    wsum = np.zeros((nx, ny, nz), dtype=float)

    # 向量化 scatter：把 8 个邻居一次性累加
    for i in range(8):
        dx, dy, dz = offset[i]
        ix = idx0[:, 0] + dx
        iy = idx0[:, 1] + dy
        iz = idx0[:, 2] + dz
        # 边界保护
        mask = (ix < nx) & (iy < ny) & (iz < nz)
        wi = np.squeeze(w[i, mask])
        # print(wi.shape,ix.shape,iy.shape,iz.shape,vals.shape) # (N,) (N,) (N,) (N,) (N,)
        np.add.at(out, (ix[mask], iy[mask], iz[mask]), wi * vals[mask])
        np.add.at(wsum, (ix[mask], iy[mask], iz[mask]), wi)

    return out

def save_mat(path,gaussian,min_pos,grid_size,pixels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    rho=np.zeros((pixels[0],pixels[1],pixels[2]),dtype=np.float32)
    o=np.zeros((pixels[0],pixels[1],pixels[2],gaussian.opacities.shape[1]),dtype=np.float32)
    c=np.zeros((pixels[0],pixels[1],pixels[2]),dtype=np.float32)
    scale=np.zeros((pixels[0],pixels[1],pixels[2]),dtype=np.float32)
    
    # 根据gaussian.means和min_pos,max_pos,grid_size，计算出每个gaussian的位置索引
    means=gaussian.means.detach().cpu().numpy()
    indices_x = np.clip(np.floor((means[:, 0] - min_pos[0]) / grid_size[0]).astype(np.int32), 0, pixels[0]-1)
    indices_y = np.clip(np.floor((means[:, 1] - min_pos[1]) / grid_size[1]).astype(np.int32), 0, pixels[1]-1)
    indices_z = np.clip(np.floor((means[:, 2] - min_pos[2]) / grid_size[2]).astype(np.int32), 0, pixels[2]-1)
    indices = (indices_x, indices_y, indices_z)
    rho[indices] = gaussian.get_colour.detach().cpu().numpy().flatten()
    o[indices] = gaussian.get_opacity.detach().cpu().numpy().reshape(-1,gaussian.opacities.shape[1])
    c[indices] = gaussian.get_coefficient.detach().cpu().numpy().flatten()
    scale[indices] = gaussian.get_scaling.detach().cpu().numpy().flatten()
    
    # means=gaussian.means.detach().cpu().numpy()
    # rho = gaussian.get_colour.detach().cpu().numpy().flatten()
    # o=gaussian.get_opacity.detach().cpu().numpy()[:,0]
    # c=gaussian.get_coefficient.detach().cpu().numpy().flatten()
    # scale=gaussian.get_scaling.detach().cpu().numpy().flatten()

    # rho=scatter_trilinear(means, rho, min_pos, grid_size, pixels)
    # c=scatter_trilinear(means, c, min_pos, grid_size, pixels)
    # scale=scatter_trilinear(means, scale, min_pos, grid_size, pixels)
    # o=scatter_trilinear(means, o, min_pos, grid_size, pixels)

    scipy.io.savemat(path,{"rho":rho,"opacity":o,"c":c,"scale":scale})

def construct_list_of_attributes(colours,scaling,rotation,features_rest,opacity):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(colours.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(features_rest.shape[1]):
            l.append('f_rest_{}'.format(i))
        for i in range(scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(opacity.shape[1]):
            l.append('opacity_{}'.format(i))
        return l

def save_ply(path,gaussian):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError as error:
        print("Already has the directory!")
    
    max_sh_degree=3 # 球谐分量的部分
    f_rest_dim=3 * (max_sh_degree + 1) ** 2 - 3

    N=gaussian.means.shape[0]

    xyz = gaussian.means.detach().cpu().numpy() # [N,3]
    normals = np.zeros_like(xyz) # [N,3]
    if gaussian.colours.shape[-1]==3:
        f_dc = gaussian.colours.detach().contiguous().cpu().numpy() # [N,3]
    else:
        f_dc = gaussian.colours.detach().contiguous().repeat(1,3).cpu().numpy()
    f_rest = np.zeros((f_dc.shape[0],f_rest_dim)) # 球谐分量, [N,f_rest_dim]
    opacities = gaussian.opacities.detach().cpu().numpy() # [N,view_num]
    scale = gaussian.scales.detach().cpu().numpy() # [N,1]
    rotation = np.zeros((N,4)) # [N,4]
    rotation[:,3]=1.0
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(f_dc,scale,rotation,f_rest,opacities)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, scale, rotation,opacities), axis=1)
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
    
    opacity_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacity_")]
    opacity_names = sorted(opacity_names, key = lambda x: int(x.split('_')[-1]))
    opacities = np.zeros((xyz.shape[0], len(opacity_names)))
    for idx, attr_name in enumerate(opacity_names):
        opacities[:, idx] = np.asarray(plydata.elements[0][attr_name])

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
    # dc_colours = np.maximum(dc_vals * SH_C0 + 0.5, np.zeros_like(dc_vals))

    output = {
        "xyz": xyz, "rot": rots, "scale": scales,
        "sh": shs, "opacity": opacities, "dc_colours": dc_vals
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
