import math
import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, Optional
from pytorch3d.ops.knn import knn_points
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.renderer.cameras import PerspectiveCameras,FoVPerspectiveCameras
from data_utils import load_gaussians_from_ply, colours_from_spherical_harmonics,unproject_depth_image
import scipy

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

class Gaussians:

    def __init__(
        self, init_type: str, device: str, load_path: Optional[str] = None,
        num_points: Optional[int] = None, isotropic: Optional[bool] = None,
        colour_dim=3,extent=1.0,center=(0,0,0)
    ):

        self.device = device
        self.colour_dim=colour_dim

        # 激活函数
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        if init_type == "gaussians":
            if isotropic is not None:
                raise ValueError((
                    "Isotropy/Anisotropy will be determined from pre-trained gaussians. "
                    "Please set isotropic to None."
                ))
            if load_path is None:
                raise ValueError

            data, is_isotropic = self._load_gaussians(load_path)
            self.is_isotropic = is_isotropic

        elif init_type == "points":
            if isotropic is not None and type(isotropic) is not bool:
                raise TypeError("isotropic must be either None or True or False.")
            if load_path is None:
                raise ValueError

            if isotropic is None:
                self.is_isotropic = False
            else:
                self.is_isotropic = isotropic

            data = self._load_points(load_path)

        elif init_type == "random":
            if isotropic is not None and type(isotropic) is not bool:
                raise TypeError("isotropic must be either None or True or False.")
            if num_points is None:
                raise ValueError

            if isotropic is None:
                self.is_isotropic = False
            else:
                self.is_isotropic = isotropic

            data = self._load_random(num_points,extent,center)

        else:
            raise ValueError(f"Invalid init_type: {init_type}")

        # 优化变量
        self.pre_act_quats = data["pre_act_quats"]
        self.means = data["means"]
        self.pre_act_scales = data["pre_act_scales"]
        self.colours = data["colours"]
        self.pre_act_opacities = data["pre_act_opacities"]

        # 优化相关
        self.optimizer_type = "default"
        self.xyz_gradient_accum = torch.empty(0) # training_setup函数中初始化
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0.01
        self.spatial_lr_scale = 1.0 # 和空间大小有关的参数

        self.sphere=False # 是否使用球谐函数

        if self.device!="cpu":
            self.to_cuda()

    def __len__(self):
        return len(self.means)

    def _load_gaussians(self, ply_path: str):

        data = dict()
        ply_gaussians = load_gaussians_from_ply(ply_path)

        data["means"] = torch.tensor(ply_gaussians["xyz"])
        data["pre_act_quats"] = torch.tensor(ply_gaussians["rot"])
        data["pre_act_scales"] = torch.tensor(ply_gaussians["scale"])
        data["pre_act_opacities"] = torch.tensor(ply_gaussians["opacity"]).squeeze()
        data["colours"] = torch.tensor(ply_gaussians["dc_colours"])

        if self.colour_dim==1:
            data["colours"]=data["colours"][:,0:1]

        is_isotropic = False
        if data["pre_act_scales"].shape[1] != 3:
            is_isotropic=True

        # 计算场景大致范围
        self.center=torch.mean(data["means"],dim=0)
        _range=torch.max(data["means"],dim=0)[0]-torch.min(data["means"],dim=0)[0]
        self.radius=torch.max(_range/2.0).to(self.device)

        return data, is_isotropic

    def _load_points(self, path: str):

        data = dict()

        if path[-4:]==".npy":
            means = np.load(path)
        elif path[-4:]==".mat":
            means = scipy.io.loadmat(path)["points"]

        # Initializing means using the provided point cloud
        data["means"] = torch.tensor(means.astype(np.float32))  # (N, 3)

        # Initializing opacities such that all when sigmoid is applied to pre_act_opacities,
        # we will have a opacity value close to (but less than) 1.0
        data["pre_act_opacities"] = 8.0 * torch.ones((len(means),), dtype=torch.float32)  # (N,)

        # Initializing colors randomly
        data["colours"] = torch.rand((len(means), self.colour_dim), dtype=torch.float32)  # (N, colour_dim)

        # Initializing quaternions to be the identity quaternion
        quats = torch.zeros((len(means), 4), dtype=torch.float32)  # (N, 4)
        quats[:, 0] = 1.0
        data["pre_act_quats"] = quats  # (N, 4)

        # Initializing scales using the mean distance of each point to its 50 nearest points
        dists, _, _ = knn_points(data["means"].unsqueeze(0), data["means"].unsqueeze(0), K=50)
        data["pre_act_scales"] = torch.log(torch.mean(dists[0], dim=1)).unsqueeze(1)  # (N, 1)

        if not self.is_isotropic:
            data["pre_act_scales"] = data["pre_act_scales"].repeat(1, 3)  # (N, 3)
        
        # 计算场景大致范围
        self.center=torch.mean(data["means"],dim=0)
        _range=torch.max(data["means"],dim=0)[0]-torch.min(data["means"],dim=0)[0]
        self.radius=torch.max(_range/2.0).to(self.device)

        return data

    def _load_random(self, num_points: int,radius=1.0,center=(0,0,0),zradius=0.2):

        data = dict()

        # Initializing means randomly
        self.center=torch.Tensor(center)
        self.radius=radius
        means_=torch.rand((num_points, 3), dtype=torch.float32) # (N, 3)
        data["means"]= radius*(means_-0.5)*2
        # data["means"][:,2]=zradius*(means_[:,2]-0.5)*2
        data["means"]+=self.center.view(1,3)

        # Initializing opacities such that all when sigmoid is applied to pre_act_opacities,
        # we will have a opacity value close to (but less than) 1.0
        data["pre_act_opacities"] = 8.0 * torch.ones((num_points,), dtype=torch.float32)  # (N,)

        # Initializing colors randomly
        data["colours"] = 0.1*torch.ones((num_points, self.colour_dim), dtype=torch.float32)  # (N, colour_dim)

        # Initializing quaternions to be the identity quaternion
        quats = torch.zeros((num_points, 4), dtype=torch.float32)  # (N, 4)
        quats[:, 0] = 1.0
        data["pre_act_quats"] = quats  # (N, 4)

        # Initializing scales randomly
        # data["pre_act_scales"] = torch.log((torch.rand((num_points, 1), dtype=torch.float32) + 1e-6) * 0.01)
        # Initializing scales using the mean distance of each point to its 50 nearest points
        dists, _, _ = knn_points(data["means"].unsqueeze(0), data["means"].unsqueeze(0), K=50)
        data["pre_act_scales"] = torch.log(torch.mean(dists[0], dim=1)).unsqueeze(1)  # (N, 1)

        if not self.is_isotropic:
            data["pre_act_scales"] = data["pre_act_scales"].repeat(1, 3)  # (N, 3)

        return data

    def _compute_jacobian(self, means_3D: torch.Tensor, camera, img_size: Tuple):

        # if camera.in_ndc():
        #     raise RuntimeError

        # fx, fy = camera.focal_length.flatten()
        W, H = img_size

        half_tan_fov_x=torch.tan(camera.fov)/2
        half_tan_fov_y=torch.tan(camera.fov)/2
        fx=W/torch.tan(camera.fov)
        fy=H/torch.tan(camera.fov)
        # half_tan_fov_x = 0.5 * W / fx
        # half_tan_fov_y = 0.5 * H / fy

        view_transform = camera.get_world_to_view_transform()
        means_view_space = view_transform.transform_points(means_3D)

        tx = means_view_space[:, 0]
        ty = means_view_space[:, 1]
        tz = means_view_space[:, 2]
        tz2 = tz*tz

        lim_x = 1.3 * half_tan_fov_x
        lim_y = 1.3 * half_tan_fov_y

        tx = torch.clamp(tx/tz, -lim_x, lim_x) * tz
        ty = torch.clamp(ty/tz, -lim_y, lim_y) * tz

        J = torch.zeros((len(tx), 2, 3))  # (N, 2, 3)
        J = J.to(self.device)

        J[:, 0, 0] = fx / tz
        J[:, 1, 1] = fy / tz
        J[:, 0, 2] = -(fx * tx) / tz2
        J[:, 1, 2] = -(fy * ty) / tz2

        return J  # (N, 2, 3)

    def check_if_trainable(self):

        attrs = ["means", "pre_act_scales", "colours", "pre_act_opacities"]
        if not self.is_isotropic:
            attrs += ["pre_act_quats"]

        for attr in attrs:
            param = getattr(self, attr)
            if not getattr(param, "requires_grad", False):
                raise Exception("Please use function make_trainable to make parameters trainable")

        if self.is_isotropic and self.pre_act_quats.requires_grad:
            raise RuntimeError("You do not need to optimize quaternions in isotropic mode.")

    def to_cuda(self):

        self.pre_act_quats = self.pre_act_quats.to(self.device)
        self.means = self.means.to(self.device)
        self.pre_act_scales = self.pre_act_scales.to(self.device)
        self.colours = self.colours.to(self.device)
        self.pre_act_opacities = self.pre_act_opacities.to(self.device)

        # [Q 1.3.1] NOTE: Uncomment spherical harmonics code for question 1.3.1
        if self.sphere:
            self.spherical_harmonics = self.spherical_harmonics.to(self.device)

    def compute_cov_3D(self, quats: torch.Tensor, scales: torch.Tensor):
        """
        Computes the covariance matrices of 3D Gaussians using equation (6) of the 3D
        Gaussian Splatting paper.

        Link: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf

        Args:
            quats   :   A torch.Tensor of shape (N, 4) representing the rotation
                        components of 3D Gaussians in quaternion form.
            scales  :   If self.is_isotropic is True, scales is will be a torch.Tensor of shape (N, 1)
                        If self.is_isotropic is False, scales is will be a torch.Tensor of shape (N, 3).
                        Represents the scaling components of the 3D Gaussians.

        Returns:
            cov_3D  :   A torch.Tensor of shape (N, 3, 3)
        """
        # NOTE: While technically you can use (almost) the same code for the
        # isotropic and anisotropic case, can you think of a more efficient
        # code for the isotropic case?

        # HINT: Are quats ever used or optimized for isotropic gaussians? What will their value be?
        # Based on your answers, can you write a more efficient code for the isotropic case?
        if self.is_isotropic:
            # 如果是各向同性，则高斯就是一个球形
            ### YOUR CODE HERE ###
            scales=scales*scales
            scales=scales.repeat((1,3))
            cov_3D=torch.diag_embed(scales) # (N, 3, 3)

        # HINT: You can use a function from pytorch3d to convert quaternions to rotation matrices.
        else:
            ### YOUR CODE HERE ###
            R=quaternion_to_matrix(quats) # (N,3,3)
            scales=scales*scales
            cov_3D=torch.diag_embed(scales) # (N, 3, 3)
            cov_3D = torch.bmm(R,torch.bmm(cov_3D,R.transpose(1,2)))  # (N, 3, 3)

        return cov_3D

    def compute_cov_2D(
        self, means_3D: torch.Tensor, quats: torch.Tensor, scales: torch.Tensor,
        camera, img_size: Tuple
    ):
        """
        Computes the covariance matrices of 2D Gaussians using equation (5) of the 3D
        Gaussian Splatting paper.

        Link: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf

        Args:
            quats       :   A torch.Tensor of shape (N, 4) representing the rotation
                            components of 3D Gaussians in quaternion form.
            scales      :   If self.is_isotropic is True, scales is will be a torch.Tensor of shape (N, 1)
                            If self.is_isotropic is False, scales is will be a torch.Tensor of shape (N, 3)
            camera      :   A pytorch3d Camera object
            img_size    :   A tuple representing the (width, height) of the image

        Returns:
            cov_3D  :   A torch.Tensor of shape (N, 3, 3)
        """
        N=quats.shape[0]
        ### YOUR CODE HERE ###
        # HINT: For computing the jacobian J, can you find a function in this file that can help?
        J = self._compute_jacobian(means_3D,camera,img_size)  # (N, 2, 3)

        ### YOUR CODE HERE ###
        # HINT: Can you extract the world to camera rotation matrix (W) from one of the inputs
        # of this function?
        W = camera.R.repeat((N,1,1))  # (N, 3, 3)

        ### YOUR CODE HERE ###
        # HINT: Can you find a function in this file that can help?
        cov_3D = self.compute_cov_3D(quats,scales)  # (N, 3, 3)

        ### YOUR CODE HERE ###
        # HINT: Use the above three variables to compute cov_2D
        J=torch.bmm(J,W)
        cov_2D = torch.bmm(J,torch.bmm(cov_3D,J.transpose(1,2)))  # (N, 2, 2)

        # Post processing to make sure that each 2D Gaussian covers atleast approximately 1 pixel
        cov_2D[:, 0, 0] += 0.3
        cov_2D[:, 1, 1] += 0.3

        return cov_2D

    @staticmethod
    def compute_means_2D(means_3D: torch.Tensor, camera):
        """
        Computes the means of the projected 2D Gaussians given the means of the 3D Gaussians.

        Args:
            means_3D    :   A torch.Tensor of shape (N, 3) representing the means of
                            3D Gaussians.
            camera      :   A pytorch3d Camera object.

        Returns:
            means_2D    :   A torch.Tensor of shape (N, 2) representing the means of
                            2D Gaussians.
        """
        ### YOUR CODE HERE ###
        # HINT: Do note that means_2D have units of pixels. Hence, you must apply a
        # transformation that moves points in the world space to screen space.
        # view_transform = camera.get_world_to_view_transform() # 从世界坐标系到相机坐标系的3D变换
        # view_transform=camera.get_full_projection_transform() # 包括从世界坐标系到相机坐标系、相机坐标系到屏幕坐标系的变换
        means_2D = camera.transform_points_screen(means_3D) # (N, 3)
        return means_2D[:,:2] # (N, 2)

    @staticmethod
    def invert_cov_2D(cov_2D: torch.Tensor):
        """
        Using the formula for inverse of a 2D matrix to invert the cov_2D matrix

        Args:
            cov_2D          :   A torch.Tensor of shape (N, 2, 2)

        Returns:
            cov_2D_inverse  :   A torch.Tensor of shape (N, 2, 2)
        """
        determinants = cov_2D[:, 0, 0] * cov_2D[:, 1, 1] - cov_2D[:, 1, 0] * cov_2D[:, 0, 1]
        determinants = determinants[:, None, None]  # (N, 1, 1)

        cov_2D_inverse = torch.zeros_like(cov_2D)  # (N, 2, 2)
        cov_2D_inverse[:, 0, 0] = cov_2D[:, 1, 1]
        cov_2D_inverse[:, 1, 1] = cov_2D[:, 0, 0]
        cov_2D_inverse[:, 0, 1] = -1.0 * cov_2D[:, 0, 1]
        cov_2D_inverse[:, 1, 0] = -1.0 * cov_2D[:, 1, 0]

        cov_2D_inverse = (1.0 / determinants) * cov_2D_inverse

        return cov_2D_inverse

    @staticmethod
    def evaluate_gaussian_2D(points_2D: torch.Tensor, means_2D: torch.Tensor, cov_2D_inverse: torch.Tensor):
        """
        Computes the exponent (power) of 2D Gaussians.

        Args:
            points_2D       :   A torch.Tensor of shape (1, H*W, 2) containing the x, y points
                                corresponding to every pixel in an image. See function
                                compute_alphas in the class Scene to get more information
                                about how points_2D is created.
            means_2D        :   A torch.Tensor of shape (N, 1, 2) representing the means of
                                N 2D Gaussians.
            cov_2D_inverse  :   A torch.Tensor of shape (N, 2, 2) representing the
                                inverse of the covariance matrices of N 2D Gaussians.

        Returns:
            power           :   A torch.Tensor of shape (N, H*W) representing the computed
                                power of the N 2D Gaussians at every pixel location in an image.
        """
        ### YOUR CODE HERE ###
        # HINT: Refer to README for a relevant equation
        vec=points_2D-means_2D # (N,H*W,2)
        vec2=torch.bmm(cov_2D_inverse,vec.transpose(1,2)).transpose(1,2) # (N,H*W,2)
        power =vec[:,:,0]*vec2[:,:,0]+vec[:,:,1]*vec2[:,:,1]  # (N, H*W)

        return -0.5*power

    @staticmethod
    def apply_activations(pre_act_quats, pre_act_scales, pre_act_opacities):

        # Convert logscales to scales
        scales = torch.exp(pre_act_scales)

        # Normalize quaternions
        quats = torch.nn.functional.normalize(pre_act_quats)

        # Bound opacities between (0, 1)
        opacities = torch.sigmoid(pre_act_opacities)

        return quats, scales, opacities
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self.pre_act_scales)
    @property
    def get_opacity(self):
        return self.opacity_activation(self.pre_act_opacities)
    @property
    def get_xyz(self):
        return self.means
    @property
    def get_colour(self):
        return self.colours**2
    
    def densify_and_clone1(self,copy_num=2,std_multiple=1):
        # 找到满足条件的片元，然后复制两份
        stds = std_multiple*self.get_scaling.repeat(copy_num,1)
        means =torch.zeros((stds.size(0), 3),device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots=quaternion_to_matrix(self.pre_act_quats.view(-1,4)).repeat(copy_num,1,1)
        # 拷贝后片元中心略有平移
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz.repeat(copy_num, 1)
        # 拷贝后scale下降
        new_scaling = self.scaling_inverse_activation(self.get_scaling.repeat(copy_num,1) / (0.8*copy_num))
        # 拷贝后旋转、颜色、透明度不变
        new_rotation = self.pre_act_quats.repeat(copy_num,1)
        new_colours = self.colours.repeat(copy_num,1)
        new_opacities = self.pre_act_opacities.repeat(copy_num)

        self.densification_postfix(new_xyz, new_colours, new_opacities, new_scaling, new_rotation)

    def density_and_split1(self,selected_pts_mask,copy_num=2,save_old=True):
        stds = self.get_scaling[selected_pts_mask].repeat(copy_num,1)
        means =torch.zeros((stds.size(0), 3),device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots=quaternion_to_matrix(self.pre_act_quats[selected_pts_mask].view(-1,4)).repeat(copy_num,1,1)
        
        # 用于替代之前的片元
        new_xyz_=self.get_xyz[selected_pts_mask]
        new_scaling_ = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask] / (0.7*copy_num)) # 拷贝后scale下降
        new_rotation_=self.pre_act_quats[selected_pts_mask]
        new_colours_ = self.colours[selected_pts_mask]
        new_opacity_ = self.pre_act_opacities[selected_pts_mask]

        # 拷贝后片元中心略有平移
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + new_xyz_.repeat(copy_num, 1)
        new_scaling=new_scaling_.repeat(copy_num,1)
        # 拷贝后旋转、颜色、透明度不变
        new_rotation = new_rotation_.repeat(copy_num,1)
        new_colours = new_colours_.repeat(copy_num,1)
        new_opacity=new_opacity_.repeat(copy_num)

        # 将新的split产生的tensor和之前的tensor合并到一起
        self.densification_postfix(new_xyz, new_colours, new_opacity, new_scaling, new_rotation)
        # 再把分裂前的tensor删除掉
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(copy_num * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(prune_filter)
        
        # 之前的位置依然保留，但是大小下降
        if save_old:
            self.densification_postfix(new_xyz_, new_colours_, new_opacity_, new_scaling_, new_rotation_)
    
    def set_scale(self,selected_pts_mask,new_scale):
        # 用于替代之前的片元
        new_xyz_=self.get_xyz[selected_pts_mask]
        new_scaling_ = self.scaling_inverse_activation(torch.ones_like(self.get_scaling[selected_pts_mask],device=self.device)*new_scale) # 拷贝后scale下降
        new_rotation_=self.pre_act_quats[selected_pts_mask]
        new_colours_ = self.colours[selected_pts_mask]
        new_opacity_ = self.pre_act_opacities[selected_pts_mask]

        # 再把分裂前的tensor删除掉
        self.prune_points(selected_pts_mask)
        # 之前的位置依然保留，但是大小下降
        self.densification_postfix(new_xyz_, new_colours_, new_opacity_, new_scaling_, new_rotation_)

    def training_setup(self, training_args):
        self.percent_dense = 0.01
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        l = [
            {'params': [self.means], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self.colours], 'lr': training_args.feature_lr, "name": "colours"},
            {'params': [self.pre_act_opacities], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self.pre_act_scales], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self.pre_act_quats], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    
    # 对高斯片元实现增删
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self.means = optimizable_tensors["xyz"]
        self.colours = optimizable_tensors["colours"]
        self.pre_act_opacities = optimizable_tensors["opacity"]
        self.pre_act_scales = optimizable_tensors["scaling"]
        self.pre_act_quats = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_colours, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "colours": new_colours,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        # 将新生成的tensor拼接到之前的变量上
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        # 然后重新赋值
        self.means = optimizable_tensors["xyz"]
        self.colours = optimizable_tensors["colours"]
        self.pre_act_opacities = optimizable_tensors["opacity"]
        self.pre_act_scales = optimizable_tensors["scaling"]
        self.pre_act_quats = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, copy_num=2):
        # print(grads.shape) # [N,1]
        n_init_points = self.get_xyz.shape[0]
        # 首先梯度要超过阈值，但是因为有一部分tensor刚刚被添加上去，因此需要选出有用的梯度
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # print(selected_pts_mask.shape) # [N]

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # print(selected_pts_mask.shape) # [mask_N]
        print("Split number:",torch.sum(selected_pts_mask).item())
        # 找到满足条件的片元，然后复制两份
        stds = self.get_scaling[selected_pts_mask].repeat(copy_num,1)
        means =torch.zeros((stds.size(0), 3),device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots=quaternion_to_matrix(self.pre_act_quats[selected_pts_mask].view(-1,4)).repeat(copy_num,1,1)
        # 拷贝后片元中心略有平移
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(copy_num, 1)
        # 拷贝后scale下降
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(copy_num,1) / (0.8*copy_num))
        # 拷贝后旋转、颜色、透明度不变
        new_rotation = self.pre_act_quats[selected_pts_mask].repeat(copy_num,1)
        new_colours = self.colours[selected_pts_mask].repeat(copy_num,1)
        new_opacity = self.pre_act_opacities[selected_pts_mask].repeat(copy_num)

        # 将新的split产生的tensor和之前的tensor合并到一起
        self.densification_postfix(new_xyz, new_colours, new_opacity, new_scaling, new_rotation)
        # 再把分裂前的tensor删除掉
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(copy_num * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # 首先梯度要超过阈值
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        print("Clone number:",torch.sum(selected_pts_mask).item())
        new_xyz = self.means[selected_pts_mask]#+1e-4*grads[selected_pts_mask]
        new_colours = self.colours[selected_pts_mask]
        new_opacities = self.pre_act_opacities[selected_pts_mask]
        new_scaling = self.pre_act_scales[selected_pts_mask]
        new_rotation = self.pre_act_quats[selected_pts_mask]

        self.densification_postfix(new_xyz, new_colours, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, grad_threshold, min_opacity, extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        print(torch.max(grads),torch.mean(grads))

        # 按照位置梯度在增加点云密度
        self.densify_and_clone(grads, grad_threshold, extent)
        self.densify_and_split(grads, grad_threshold, extent)

        # 透明度小于阈值的直接裁剪
        prune_mask1 = (self.get_opacity < min_opacity).squeeze()
        # prune_mask2 = (self.get_colour < 1e-4).squeeze()
        prune_mask2=self.get_scaling.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(prune_mask1,prune_mask2)
        # # 投影后太大的片元要被删除
        # if max_screen_size:
        #     big_points_vs = self.max_radii2D > max_screen_size
        #     big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        #     prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        # print("Prune number:",torch.sum(prune_mask).item())
        # self.prune_points(prune_mask)

        # tmp_radii = self.tmp_radii
        # self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

class Scene:

    def __init__(self, gaussians: Gaussians):
        self.gaussians = gaussians
        self.device = self.gaussians.device

    def __repr__(self):
        return f"<Scene with {len(self.gaussians)} Gaussians>"

    def compute_depth_values(self, camera):
        """
        Computes the depth value of each 3D Gaussian.

        Args:
            camera  :   A pytorch3d Camera object.

        Returns:
            z_vals  :   A torch.Tensor of shape (N,) with the depth of each 3D Gaussian.
        """
        ### YOUR CODE HERE ###
        # HINT: You can use get the means of 3D Gaussians self.gaussians and calculate
        # the depth using the means and the camera

        means_2D = camera.get_world_to_view_transform().transform_points(self.gaussians.means) # 世界坐标系转相机坐标系
        # means_2D = camera.transform_points(self.gaussians.means) # (N, 3) # 世界坐标系转screen坐标系，xy不是真实坐标，而是是像素位置，而z毫无意义

        z_vals = means_2D[:,2]  # (N,)

        return z_vals

    def get_idxs_to_filter_and_sort(self, z_vals: torch.Tensor):
        """
        Given depth values of Gaussians, return the indices to depth-wise sort
        Gaussians and at the same time remove invalid Gaussians.

        You can see the function render to see how the returned indices will be used.
        You are required to create a torch.Tensor idxs such that by using them in the
        function render we can arrange Gaussians (or equivalently their attributes such as
        the mean) in ascending order of depth. You should also make sure to not include indices
        that correspond to Gaussians with depth value less than 0.

        idxs should be torch.Tensor of dtype int64 with length N (N <= M, where M is the
        total number of Gaussians before filtering)

        Please refer to the README file for more details.
        """
        ### YOUR CODE HERE ###
        idxs = torch.argsort(z_vals)  # (M,)

        # 统计小于0的元素的个数
        count = torch.sum(z_vals<=0)

        return idxs[count:]

    def compute_exp_power(self, means_2D, cov_2D, img_size):
        """
        Given some parameters of N ordered Gaussians, this function computes
        the alpha values.

        Args:
            means_2D    :   A torch.Tensor of shape (N, 2) with the means
                            of the 2D Gaussians.
            cov_2D      :   A torch.Tensor of shape (N, 2, 2) with the covariances
                            of the 2D Gaussians.
            img_size    :   The (width, height) of the image to be rendered.


        Returns:
            alphas      :   A torch.Tensor of shape (N, H, W) with the computed alpha
                            values for each of the N ordered Gaussians at every
                            pixel location.
        """
        W, H = img_size

        # point_2D contains all possible pixel locations in an image
        xs, ys = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
        points_2D = torch.stack((xs.flatten(), ys.flatten()), dim = 1)  # (H*W, 2)
        points_2D = points_2D.to(self.device)

        points_2D = points_2D.unsqueeze(0)  # (1, H*W, 2)
        means_2D = means_2D.unsqueeze(1)  # (N, 1, 2)

        ### YOUR CODE HERE ###
        # HINT: Can you find a function in this file that can help?
        cov_2D_inverse = Gaussians.invert_cov_2D(cov_2D)  # (N, 2, 2) TODO: Verify shape

        ### YOUR CODE HERE ###
        # HINT: Can you find a function in this file that can help?
        power = Gaussians.evaluate_gaussian_2D(points_2D,means_2D,cov_2D_inverse)  # (N, H*W)

        # Computing exp(power) with some post processing for numerical stability
        exp_power = torch.where(power > 0.0, 0.0, torch.exp(power))

        exp_power = torch.reshape(exp_power, (-1, H, W))  # (N, H, W)

        return exp_power

    def compute_alphas(self, opacities, exp_power):
        """
        Given some parameters of N ordered Gaussians, this function computes
        the alpha values.

        Args:
            opacities   :   A torch.Tensor of shape (N,) with the opacity value
                            of each Gaussian.
            exp_power    :  A torch.Tensor of shape (N, H, W)

        Returns:
            alphas      :   A torch.Tensor of shape (N, H, W) with the computed alpha
                            values for each of the N ordered Gaussians at every
                            pixel location.
        """
        N, W, H = exp_power.shape

        ### YOUR CODE HERE ###
        # HINT: Refer to README for a relevant equation.
        alphas = opacities.unsqueeze(1).unsqueeze(1)*exp_power  # (N, H, W)
    
        # Post processing for numerical stability
        alphas = torch.minimum(alphas, torch.full_like(alphas, 0.99))
        alphas = torch.where(alphas < 1/255.0, 0.0, alphas)

        return alphas

    def compute_transmittance(
        self, alphas: torch.Tensor,
        start_transmittance: Optional[torch.Tensor] = None
    ):
        """
        Given the alpha values of N ordered Gaussians, this function computes
        the transmittance.

        The variable start_transmittance contains information about the transmittance
        at each pixel location BEFORE encountering the first Gaussian in the input.
        This variable is useful when computing transmittance in mini-batches because
        we would require information about the transmittance accumulated until the
        previous mini-batch to begin computing the transmittance for the current mini-batch.

        In case there were no previous mini-batches (or we are splatting in one-shot
        without using mini-batches), then start_transmittance will be None (since no Gaussians
        have been encountered so far). In this case, the code will use a starting
        transmittance value of 1.

        Args:
            alphas                  :   A torch.Tensor of shape (N, H, W) with the computed alpha
                                        values for each of the N ordered Gaussians at every
                                        pixel location.
            start_transmittance     :   Can be None or a torch.Tensor of shape (1, H, W). Please
                                        see the docstring for more information.

        Returns:
            transmittance           :   A torch.Tensor of shape (N, H, W) with the computed transmittance
                                        values for each of the N ordered Gaussians at every
                                        pixel location.
        """
        _, H, W = alphas.shape

        if start_transmittance is None:
            S = torch.ones((1, H, W), device=alphas.device, dtype=alphas.dtype)
        else:
            S = start_transmittance

        one_minus_alphas = 1.0 - alphas
        one_minus_alphas = torch.concat((S, one_minus_alphas), dim=0)  # (N+1, H, W)

        ### YOUR CODE HERE ###
        # HINT: Refer to README for a relevant equation.
        transmittance = torch.cumprod(one_minus_alphas,dim=0)  
        transmittance=transmittance[:-1,:,:] # (N, H, W)
        # 因为透过率只计算到自己的前一个位置的乘积，所以要减去最后一个位置，不然会得到错误结果，这个bug我找了好久才发现

        # Post processing for numerical stability
        transmittance = torch.where(transmittance < 1e-4, 0.0, transmittance)  # (N, H, W)

        return transmittance

    def splat(
        self, camera, means_3D: torch.tensor, z_vals: torch.Tensor,
        quats: torch.Tensor, scales: torch.Tensor, colours: torch.Tensor,
        opacities: torch.Tensor, img_size: Tuple = (256, 256),
        start_transmittance: Optional[torch.Tensor] = None
    ):
        """
        Given N ordered (depth-sorted) 3D Gaussians (or equivalently in our case,
        the parameters of the 3D Gaussians like means, quats etc.), this function splats
        them to the image plane to render an RGB image, depth map and a silhouette map.

        Args:
            camera                  :   A pytorch3d Camera object.
            means_3D                :   A torch.Tensor of shape (N, 3) with the means
                                        of the 3D Gaussians.
            z_vals                  :   A torch.Tensor of shape (N,) with the depths
                                        of the 3D Gaussians. # TODO: Verify shape
            quats                   :   A torch.Tensor of shape (N, 4) representing the rotation
                                        components of 3D Gaussians in quaternion form.
            scales                  :   A torch.Tensor of shape (N, 1) (if isotropic) or
                                        (N, 3) (if anisotropic) representing the scaling
                                        components of 3D Gaussians.
            colours                 :   A torch.Tensor of shape (N, colour_dim) with the colour contribution
                                        of each Gaussian.
            opacities               :   A torch.Tensor of shape (N,) with the opacity of each Gaussian.
            img_size                :   The (width, height) of the image.
            start_transmittance     :   Please see the docstring of the function compute_transmittance
                                        for information about this argument.

        Returns:
            image                   :   A torch.Tensor of shape (H, W, 3) with the rendered RGB colour image.
            depth                   :   A torch.Tensor of shape (H, W, 1) with the rendered depth map.
            mask                    :   A torch.Tensor of shape (H, W, 1) with the rendered silhouette map.
            final_transmittance     :   A torch.Tensor of shape (1, H, W) representing the transmittance at
                                        each pixel computed using the N ordered Gaussians. This will be useful
                                        for mini-batch splatting in the next iteration.
        """
        N=means_3D.shape[0]
        if z_vals.shape[0]!=N:
            raise RuntimeError
        # Step 1: Compute 2D gaussian parameters

        ### YOUR CODE HERE ###
        # HINT: Can you find a function in this file that can help?
        means_2D = Gaussians.compute_means_2D(means_3D,camera)  # (N, 2)
        # if not no_grad:
        #     means_2D.retain_grad()

        ### YOUR CODE HERE ###
        # HINT: Can you find a function in this file that can help?
        cov_2D = self.gaussians.compute_cov_2D(means_3D,quats,scales,camera,img_size)  # (N, 2, 2)

        # # 计算每个片元投影后在图片上的半径（2*2协方差矩阵的特征值）
        # mid = 0.5*(cov_2D[:,0,0] + cov_2D[:,1,1])
        # det = cov_2D[:,0,0] * cov_2D[:,1,1] - cov_2D[:,1,0]*cov_2D[:,0,1]
        # temp=torch.Tensor([0.1]).to(det.device)
        # lambda1 = mid + torch.sqrt(torch.max(temp, mid * mid - det))
        # lambda2 = mid - torch.sqrt(torch.max(temp, mid * mid - det))
        # radii = torch.ceil(3.0* torch.sqrt(torch.max(lambda1, lambda2))) # (N,1)

        # Step 2: Compute alpha maps for each gaussian

        ### YOUR CODE HERE ###
        # HINT: Can you find a function in this file that can help?
        exp_power=self.compute_exp_power(means_2D,cov_2D,img_size) # (N, H, W)
        alphas = self.compute_alphas(opacities,exp_power)  # (N, H, W)

        # Step 3: Compute transmittance maps for each gaussian

        ### YOUR CODE HERE ###
        # HINT: Can you find a function in this file that can help?
        transmittance = self.compute_transmittance(alphas,start_transmittance)  # (N, H, W)

        # Some unsqueezing to set up broadcasting for vectorized implementation.
        # You can selectively comment these out if you want to compute things
        # in a diferent way.
        z_vals = z_vals[:, None, None, None]  # (N, 1, 1, 1)
        alphas = alphas[..., None]  # (N, H, W, 1)
        colours = colours[:, None, None, :]  # (N, 1, 1, colour_dim)
        transmittance = transmittance[..., None]  # (N, H, W, 1)

        # Step 4: Create image, depth and mask by computing the colours for each pixel.

        ### YOUR CODE HERE ###
        # HINT: Refer to README for a relevant equation
        image = torch.sum(colours*alphas*transmittance,dim=0)  # (H, W, colour_dim)

        ### YOUR CODE HERE ###
        # if use alpha blend
        # depth = torch.sum(z_vals*alphas*transmittance,dim=0)  # (H, W, 1)

        # if use interaction
        m_ = (exp_power[..., None] > 0.9).float()  # [N, H, W, 1]
        # 找到第一个超过0.9的位置的索引
        first_occurrence = torch.argmax(m_, dim=0)  # [H, W, 1]
        # 创建一个掩码，标记哪些位置没有超过0.9的值
        empty_mask = ~m_.any(dim=0)  # [H, W]
        # 使用first_occurrence索引z来获取depth
        depth = z_vals[first_occurrence, 0, 0, 0]  # [H, W,1]
        depth[empty_mask] = 0

        ### YOUR CODE HERE ###
        # HINT: Can you implement an equation inspired by the equation for colour?
        # mask = torch.sum(alphas*transmittance,dim=0)  # (H, W, 1)

        final_transmittance = transmittance[-1, ..., 0].unsqueeze(0)  # (1, H, W)
        return image, depth, final_transmittance

    def render(
        self, camera,per_splat: int = -1, img_size: Tuple = (256, 256)
    ):
        """
        Given a scene represented by N 3D Gaussians, this function renders the RGB
        colour image, the depth map and the silhouette map that can be observed
        from a given pytorch 3D camera.

        Args:
            camera      :   A pytorch3d Cameras object.
            per_splat   :   Number of gaussians to splat in one function call. If set to -1,
                            then all gaussians in the scene are splat in a single function call.
                            If set to any other positive interger, then it determines the number of
                            gaussians to splat per function call (the last function call might splat
                            lesser number of gaussians). In general, the algorithm can run faster
                            if more gaussians are splat per function call, but at the cost of higher GPU
                            memory consumption.
            img_size    :   The (width, height) of the image to be rendered.

        Returns:
            image       :   A torch.Tensor of shape (H, W, 3) with the rendered RGB colour image.
            depth       :   A torch.Tensor of shape (H, W, 1) with the rendered depth map.
        """
        # Globally sort gaussians according to their depth value
        z_vals_origin = self.compute_depth_values(camera) # (N,)
        idxs = self.get_idxs_to_filter_and_sort(z_vals_origin)

        pre_act_quats = self.gaussians.pre_act_quats[idxs]
        pre_act_scales = self.gaussians.pre_act_scales[idxs]
        pre_act_opacities = self.gaussians.pre_act_opacities[idxs]
        z_vals = z_vals_origin[idxs]
        means_3D = self.gaussians.means[idxs]

        # For questions 1.1, 1.2 and 1.3.2, use the below line of code for colours.
        colours = self.gaussians.get_colour[idxs] # 不用球谐分量

        # Apply activations
        quats, scales, opacities = self.gaussians.apply_activations(
            pre_act_quats, pre_act_scales, pre_act_opacities
        )

        if per_splat == -1:
            num_mini_batches = 1
        elif per_splat > 0:
            num_mini_batches = math.ceil(len(means_3D) / per_splat)
        else:
            raise ValueError("Invalid setting of per_splat")

        # In this case we can directly splat all gaussians onto the image
        if num_mini_batches == 1:

            # Get image, depth and mask via splatting
            image, depth,  _= self.splat(
                camera, means_3D, z_vals, quats, scales,
                colours, opacities, img_size
            )

        # In this case we splat per_splat number of gaussians per iteration. This makes
        # the implementation more memory efficient but at the same time makes it slower.
        else:

            W, H = img_size
            D = means_3D.device
            start_transmittance = torch.ones((1, H, W), dtype=torch.float32).to(D)
            image = torch.zeros((H, W, self.gaussians.colour_dim), dtype=torch.float32).to(D)
            depth = torch.zeros((H, W, 1), dtype=torch.float32).to(D)

            # 每次计算一部分片元的颜色，最后叠加起来得到总的结果
            for b_idx in range(num_mini_batches):

                quats_ = quats[b_idx * per_splat: (b_idx+1) * per_splat]
                scales_ = scales[b_idx * per_splat: (b_idx+1) * per_splat]
                z_vals_ = z_vals[b_idx * per_splat: (b_idx+1) * per_splat]
                colours_ = colours[b_idx * per_splat: (b_idx+1) * per_splat]
                means_3D_ = means_3D[b_idx * per_splat: (b_idx+1) * per_splat]
                opacities_ = opacities[b_idx * per_splat: (b_idx+1) * per_splat]

                # Get image, depth and mask via splatting
                image_, depth_, start_transmittance= self.splat(
                    camera, means_3D_, z_vals_, quats_, scales_,
                    colours_, opacities_, img_size,start_transmittance
                )

                image = image + image_

                # if use alpha blend
                # depth = depth + depth_ 

                # if use interaction
                xor_mask=(depth==0)^(depth_==0)
                minvalue=torch.minimum(depth,depth_)
                maxvalue=torch.maximum(depth,depth_)
                depth=torch.where(xor_mask,maxvalue,minvalue)

        return image, depth

    def calculate_gaussian_directions(self, means_3D, camera):
        """
        [Q 1.3.1] Calculates the world frame direction vectors that point from the
        camera's origin to each 3D Gaussian.

        Args:
            means_3D        :   A torch.Tensor of shape (N, 3) with the means
                                of the 3D Gaussians.
            camera          :   A pytorch3d Camera object.

        Returns:
            gaussian_dirs   :   A torch.Tensor of shape (N, 3) representing the direction vector
                                that points from the camera's origin to each 3D Gaussian.
        """
        ### YOUR CODE HERE ###
        # HINT: Think about how to get the camera origin in the world frame.
        # HINT: Do not forget to normalize the computed directions.
        center=camera.get_camera_center() # [1,3]
        gaussian_dirs = means_3D-center  # (N, 3)
        gaussian_dirs=torch.nn.functional.normalize(gaussian_dirs, p=2, dim=1)
        return gaussian_dirs

    def render_conf_hist1(self, camera,bin_resolution,num_bins):
        z_vals = self.compute_depth_values(camera) # (N,)

        intensity=self.gaussians.get_opacity.flatten()*self.gaussians.get_colour.flatten()

        intensity=intensity/(z_vals**2)

        indices_float=z_vals*2/bin_resolution # 计算索引
        indices_float = torch.clamp(indices_float, 0, num_bins - 1).flatten()  # 防止索引超出范围

        # 强度按照距离分配到两个bin上
        indices=torch.cat([indices_float.long(),indices_float.long()+1],dim=0)
        weight=indices_float-indices_float.long()
        intensity=torch.cat([intensity*(1-weight),intensity*weight],dim=0)

        hist=torch.zeros((num_bins,),dtype=torch.float32,device=camera.device) # 这里不要写require梯度，因为这个内存要在scatter_add_的时候被占掉
        # 利用scatter_add将强度值叠加到对应bin
        hist.scatter_add_(0, indices,intensity)

        return hist
    
    # 利用极坐标的形式计算histogram
    def render_conf_hist2(self, camera,bin_resolution,num_bins):
        # 计算强度
        intensity=self.gaussians.get_opacity.flatten()*self.gaussians.get_colour.flatten() # (N,)

        # 计算片元中心的深度
        r0 = self.compute_depth_values(camera).unsqueeze(1) # (N,1)

        r_=bin_resolution/2*torch.arange(1,1+num_bins,dtype=torch.float32).to(self.device).flatten() # (M,)
        r=r_.view(1,num_bins) #(1,M)

        sigma=torch.mean(self.gaussians.get_scaling,dim=1).unsqueeze(1) # (N,1)
        sigma=torch.clip(sigma,bin_resolution/2) #一定要不小于分辨率才能保证数值稳定

        # 每个深度的概率
        pdf=math.sqrt(0.5/math.pi) * (r/(r0*sigma))*torch.exp(-0.5*((r-r0)/sigma)**2) # 概率密度,[N,M]
        pr=pdf*bin_resolution/2 # 概率, [N,M]
        pr=torch.clip(pr,0,1)

        # print(torch.mean(torch.sum(pr,1)))

        hist=intensity.unsqueeze(1)*pr # (N,M)
        hist=torch.sum(hist,dim=0).flatten()
        hist=hist/r_**2

        return hist

    def render_conf_hist(
        self, camera,bin_resolution,num_bins,
        per_splat: int = -1, img_size: Tuple = (128, 128),is_train=False # 是否进入训练模式（强度-深度解耦）
    ):
        intensity, depth=self.render(camera,per_splat,img_size)
        if intensity.shape[-1]==3:
            intensity = 0.29900 * intensity[:,:,0:1] + 0.58700 * intensity[:,:,1:2] + 0.11400 * intensity[:,:,2:3] # RGB2grey

        select_mask = torch.where(depth > 0.1, True, False) # 深度阈值应该可以调整
        hist_inten=intensity[select_mask]/(depth[select_mask]**2) # 形状是[select_num]
        indices=(depth[select_mask]*2/bin_resolution).long() # 计算索引
        indices = torch.clamp(indices, 0, num_bins - 1).flatten()  # 防止索引超出范围

        # print(hist_inten.shape)

        # 没法传递梯度
        hist=torch.zeros((num_bins,),dtype=torch.float32,device=camera.device) # 这里不要写require梯度，因为这个内存要在scatter_add_的时候被占掉
        # 利用scatter_add将强度值叠加到对应bin
        if is_train:
            hist.scatter_add_(0, indices,intensity[select_mask].flatten())
        else:
            hist.scatter_add_(0, indices, hist_inten.flatten())

        # with torch.no_grad():
        #     img = intensity.detach().cpu().numpy()
        #     depth = depth.detach().cpu().numpy()
        #     depth = depth[:, :, 0].astype(np.float32)
        #     histtt=hist.detach().cpu().numpy()
        #     scipy.io.savemat("temp/depth.mat",{"img":img,"depth":depth,"hist":histtt})
        #     exit()
        return hist

    def render_nonconf_hist1(self, detect_camera,laser_camera,bin_resolution,num_bins):
        z_vals1 = self.compute_depth_values(detect_camera) # (N,)
        z_vals2 = self.compute_depth_values(laser_camera) # (N,)

        intensity=self.gaussians.get_opacity.flatten()*self.gaussians.get_colour.flatten()

        intensity=intensity/(z_vals1*z_vals1*z_vals2*z_vals2)

        indices=((z_vals1+z_vals2)/bin_resolution).long() # 计算索引
        indices = torch.clamp(indices, 0, num_bins - 1).flatten()  # 防止索引超出范围

        hist=torch.zeros((num_bins,),dtype=torch.float32,device=detect_camera.device) # 这里不要写require梯度，因为这个内存要在scatter_add_的时候被占掉
        # 利用scatter_add将强度值叠加到对应bin
        hist.scatter_add_(0, indices,intensity)

        return hist

    def render_nonconf_hist(
        self, camera,
        laser_point: torch.Tensor, # [1,3]
        bin_resolution,num_bins,
        per_splat: int = -1, img_size: Tuple = (128, 128),
        bg_colour: Tuple = (0.0, 0.0, 0.0),
        no_grad=False
    ):
        intensity, depth=self.render(camera,per_splat,img_size,bg_colour,no_grad)
        if intensity.shape[-1]==3:
            intensity = 0.29900 * intensity[:,:,0:1] + 0.58700 * intensity[:,:,1:2] + 0.11400 * intensity[:,:,2:3] # RGB2grey
        
        select_mask = torch.where(mask > 0.5, True, False)
        d2=depth[select_mask]

        # unproject depth map to 3D points
        points=unproject_depth_image(depth,camera) #[H,W,3]
        d1=points[select_mask]-laser_point # [M,3]
        d1=torch.sqrt((d1 ** 2).sum(dim=1))
        print(torch.max(d1))
        print(torch.min(d1))

        hist_inten=intensity[select_mask]/(d2*d1) # 形状是[select_num]
        indices=((d1+d2)/bin_resolution).long() # 计算索引
        indices = torch.clamp(indices, 0, num_bins - 1).flatten()  # 防止索引超出范围

        ### 这个做法只有在所有片元都能被laser_point看到的情况下才是对的！！！

        # 没法传递梯度
        hist=torch.zeros((num_bins,),dtype=torch.float32,device=camera.device) # 这里不要写require梯度，因为这个内存要在scatter_add_的时候被占掉
        # 利用scatter_add将强度值叠加到对应bin
        hist.scatter_add_(0, indices, hist_inten.flatten())
        hist=torch.clamp(hist,0)

        return hist