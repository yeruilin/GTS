import math
import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, Optional
from pytorch3d.ops.knn import knn_points
import scipy
from data_utils import load_gaussians_from_ply

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
        colour_dim=3,extent=1.0,center=(0,0,0),scale=0.01
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

            data = self._load_random(num_points,extent,center,scale)

        else:
            raise ValueError(f"Invalid init_type: {init_type}")

        # 优化变量
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
            self.to(self.device)

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

    def _load_random(self, num_points,radius,center=(0,0,0),scale=0.01):

        data = dict()

        # Initializing means randomly
        self.center=torch.Tensor(center)
        means_=torch.rand((num_points, 3), dtype=torch.float32) # (N, 3)

        if type(radius)==type([]):
            self.radius=max(radius)
            radius=torch.tensor(radius).view(1,3)
        else:
            self.radius=radius
            
        data["means"]= radius*(means_-0.5)*2
        data["means"]+=self.center.view(1,3)

        # Initializing opacities such that all when sigmoid is applied to pre_act_opacities,
        # we will have a opacity value close to (but less than) 1.0
        data["pre_act_opacities"] = 8.0 * torch.ones((num_points,), dtype=torch.float32)  # (N,)

        # Initializing colors randomly
        data["colours"] = 0.1*torch.ones((num_points, self.colour_dim), dtype=torch.float32)  # (N, colour_dim)

        # Initializing scales randomly
        # data["pre_act_scales"] = torch.log((torch.rand((num_points, 1), dtype=torch.float32) + 1e-6) * 0.01)
        # Initializing scales using the mean distance of each point to its 50 nearest points
        dists, _, _ = knn_points(data["means"].unsqueeze(0), data["means"].unsqueeze(0), K=50)
        # data["pre_act_scales"] = torch.log(torch.mean(dists[0], dim=1)).unsqueeze(1)  # (N, 1)
        data["pre_act_scales"] = torch.log(torch.ones((num_points,1),dtype=torch.float32)*scale)  # (N, 1)

        if not self.is_isotropic:
            data["pre_act_scales"] = data["pre_act_scales"].repeat(1, 3)  # (N, 3)

        return data

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

    def to(self,device):
        self.device=device
        self.means = self.means.to(self.device)
        self.pre_act_scales = self.pre_act_scales.to(self.device)
        self.colours = self.colours.to(self.device)
        self.pre_act_opacities = self.pre_act_opacities.to(self.device)

        # [Q 1.3.1] NOTE: Uncomment spherical harmonics code for question 1.3.1
        if self.sphere:
            self.spherical_harmonics = self.spherical_harmonics.to(self.device)

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
        # 拷贝后片元中心略有平移
        new_xyz = samples + self.get_xyz.repeat(copy_num, 1)

        # # 找到最近的几个点
        # dists, idxs, _ = knn_points(self.get_xyz.unsqueeze(0), self.get_xyz.unsqueeze(0), K=copy_num)
        # neighbors = self.get_xyz[idxs.squeeze(0)]  # (N, k, 3)
        # trisection_points = (2 * self.get_xyz.unsqueeze(1) + neighbors) / 3  # (N, k, 3) # 计算靠近本侧的三等分点
        # new_xyz=trisection_points.view(-1,3)

        # 拷贝后scale下降
        new_scaling = self.scaling_inverse_activation(self.get_scaling.repeat(copy_num,1) / (0.8*copy_num))
        # 拷贝后旋转、颜色、透明度不变
        new_colours = self.colours.repeat(copy_num,1)
        new_opacities = self.pre_act_opacities.repeat(copy_num)

        self.densification_postfix(new_xyz, new_colours, new_opacities, new_scaling)

    def density_and_split1(self,selected_pts_mask,copy_num=2,save_old=True):
        stds = self.get_scaling[selected_pts_mask].repeat(copy_num,1)
        means =torch.zeros((stds.size(0), 3),device=self.device)
        samples = torch.normal(mean=means, std=stds)
        
        # 用于替代之前的片元
        new_xyz_=self.get_xyz[selected_pts_mask]
        new_scaling_ = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask] / (0.7*copy_num)) # 拷贝后scale下降
        new_colours_ = self.colours[selected_pts_mask]
        new_opacity_ = self.pre_act_opacities[selected_pts_mask]

        # 拷贝后片元中心略有平移
        new_xyz = samples + new_xyz_.repeat(copy_num, 1)
        new_scaling=new_scaling_.repeat(copy_num,1)
        # 拷贝后旋转、颜色、透明度不变
        new_colours = new_colours_.repeat(copy_num,1)
        new_opacity=new_opacity_.repeat(copy_num)

        # 将新的split产生的tensor和之前的tensor合并到一起
        self.densification_postfix(new_xyz, new_colours, new_opacity, new_scaling)
        # 再把分裂前的tensor删除掉
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(copy_num * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(prune_filter)
        
        # 之前的位置依然保留，但是大小下降
        if save_old:
            self.densification_postfix(new_xyz_, new_colours_, new_opacity_, new_scaling_)
    
    def set_scale(self,selected_pts_mask,new_scale):
        # 用于替代之前的片元
        new_xyz_=self.get_xyz[selected_pts_mask]
        new_scaling_ = self.scaling_inverse_activation(torch.ones_like(self.get_scaling[selected_pts_mask],device=self.device)*new_scale) # 拷贝后scale下降
        new_colours_ = self.colours[selected_pts_mask]
        new_opacity_ = self.pre_act_opacities[selected_pts_mask]

        # 再把分裂前的tensor删除掉
        self.prune_points(selected_pts_mask)
        # 之前的位置依然保留，但是大小下降
        self.densification_postfix(new_xyz_, new_colours_, new_opacity_, new_scaling_)

    def training_setup(self, training_args):
        self.pre_act_scales.requires_grad=True
        self.colours.requires_grad=True
        self.pre_act_opacities.requires_grad=True

        l = [
            # {'params': [self.means], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self.colours], 'lr': training_args.feature_lr, "name": "colours"},
            {'params': [self.pre_act_opacities], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self.pre_act_scales], 'lr': training_args.scaling_lr, "name": "scaling"}
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

        self.means = self.means[valid_points_mask]

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self.colours = optimizable_tensors["colours"]
        self.pre_act_opacities = optimizable_tensors["opacity"]
        self.pre_act_scales = optimizable_tensors["scaling"]

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

    def densification_postfix(self, new_xyz, new_colours, new_opacities, new_scaling):
        self.means = torch.cat([self.means,new_xyz],dim=0)

        d = {
        "colours": new_colours,
        "opacity": new_opacities,
        "scaling" : new_scaling}

        # 将新生成的tensor拼接到之前的变量上
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        # 然后重新赋值
        self.colours = optimizable_tensors["colours"]
        self.pre_act_opacities = optimizable_tensors["opacity"]
        self.pre_act_scales = optimizable_tensors["scaling"]
    
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


    # 利用极坐标的形式计算histogram
    def render_conf_hist(self, scan_point,bin_resolution,num_bins,t0,decay=2.0):
        # 计算强度
        intensity=self.gaussians.get_opacity.flatten()*self.gaussians.get_colour.flatten() # (N,)

        # 计算片元中心的深度:scan_point is [1,3]
        r0 = torch.norm(self.gaussians.means-scan_point, p=2, dim=1).unsqueeze(1) # (N,1)

        r_=t0/2+bin_resolution/2*torch.arange(1,1+num_bins,dtype=torch.float32).to(self.device).flatten() # (M,)
        r=r_.view(1,num_bins) #(1,M)

        sigma=torch.mean(self.gaussians.get_scaling,dim=1).unsqueeze(1) # (N,1)
        sigma=torch.clip(sigma,bin_resolution/2) #一定要不小于分辨率才能保证数值稳定

        # 概率密度,[N,M]
        # pdf=math.sqrt(0.5/math.pi) * (r/(r0*sigma))*torch.exp(-0.5*((r-r0)/sigma)**2) # 完整的径向分布
        pdf=math.sqrt(0.5/math.pi)*torch.exp(-0.5*((r-r0)/sigma)**2)/sigma # 高斯分布
        # pdf=(r-r0)*torch.exp(-0.5*((r-r0)/sigma)**2)/(sigma**2) # 瑞利分布
        pr=pdf*bin_resolution/2 # 概率, [N,M]
        pr=torch.clip(pr,0,1)

        # print(torch.mean(torch.sum(pr,1)))

        hist=intensity.unsqueeze(1)*pr # (N,M)
        hist=torch.sum(hist,dim=0).flatten()
        hist=hist/torch.pow(r_,decay)

        return hist
    
    # 利用极坐标的形式计算histogram
    def render_conf_hist2(self, scan_point,bin_resolution,num_bins,t0=0,decay=2.0):
        # 计算强度
        intensity=self.gaussians.get_colour.flatten() # (N,)

        # 计算两组基的系数
        coeff=self.gaussians.get_opacity.flatten().unsqueeze(1) # (N,1)

        # 计算片元中心的深度:scan_point is [1,3]
        r0 = torch.norm(self.gaussians.means-scan_point, p=2, dim=1).unsqueeze(1) # (N,1)

        r_=t0/2+bin_resolution/2*torch.arange(1,1+num_bins,dtype=torch.float32).to(self.device).flatten() # (M,)
        r=r_.view(1,num_bins) #(1,M)

        sigma=torch.mean(self.gaussians.get_scaling,dim=1).unsqueeze(1) # (N,1)
        sigma=torch.clip(sigma,bin_resolution/2) #一定要不小于分辨率才能保证数值稳定

        # 概率密度,[N,M]
        pdf1=math.sqrt(0.5/math.pi)*torch.exp(-0.5*((r-r0)/sigma)**2)/sigma # 高斯分布
        pdf2=(r-r0)*torch.exp(-0.5*((r-r0)/sigma)**2)/(sigma**2) # 瑞利分布
        pdf=coeff*pdf1+(1-coeff)*pdf2 # 两个分布叠加
        pr=pdf*bin_resolution/2 # 概率, [N,M]
        pr=torch.clip(pr,0,1)

        # print(torch.mean(torch.sum(pr,1)))

        hist=intensity.unsqueeze(1)*pr # (N,M)
        hist=torch.sum(hist,dim=0).flatten()
        hist=hist/torch.pow(r_,decay)

        return hist

    # 利用极坐标的形式计算histogram
    def render_nonconf_hist(self, laserPos,laserOrigin,cameraPos,cameraOrigin,bin_resolution,num_bins,t0=0):
        # 计算强度
        intensity=self.gaussians.get_opacity.flatten()*self.gaussians.get_colour.flatten() # (N,)

        # 计算激光点和相机点到两边的距离
        r0_=torch.norm(cameraPos-cameraOrigin, p=2, dim=1)
        r0_-=t0
        if laserOrigin!=None:
            r0_+=torch.norm(laserPos-laserOrigin, p=2, dim=1)

        # 计算片元中心的深度
        a = torch.norm(self.gaussians.means-laserPos, p=2, dim=1).unsqueeze(1) # (N,1)
        b = torch.norm(self.gaussians.means-cameraPos, p=2, dim=1).unsqueeze(1) # (N,1)
        r0=a+b+r0_ # (N,1)

        r_=bin_resolution*torch.arange(1,1+num_bins,dtype=torch.float32).to(self.device).flatten() # (M,)
        r=r_.view(1,num_bins) #(1,M)

        sigma=torch.mean(self.gaussians.get_scaling,dim=1).unsqueeze(1) # (N,1)
        sigma=torch.clip(sigma,bin_resolution/2) #一定要不小于分辨率才能保证数值稳定

        # 每个深度的概率
        pdf=math.sqrt(1/math.pi)*torch.exp(-((r-r0)**2/4/sigma**2))/2/sigma # 概率密度,[N,M]
        pr=pdf*bin_resolution
        # print(torch.sum(pr,dim=1))
        pr=torch.clip(pr,0,1)

        hist=intensity.unsqueeze(1)/((a*b)) # (N,1)
        hist=hist*pr # (N,M)
        hist=torch.sum(hist,dim=0).flatten()

        # hist[0:100]=0

        return hist

    def render_nonconf_hist2(self, laserPos,laserOrigin,cameraPos,cameraOrigin,bin_resolution,num_bins,t0=0):
        # 计算强度
        intensity=self.gaussians.get_colour.flatten() # (N,)

        # 计算两组基的系数
        coeff=self.gaussians.pre_act_opacities.flatten().unsqueeze(1) # (N,1)

        # 计算激光点和相机点到两边的距离
        r0_=torch.norm(cameraPos-cameraOrigin, p=2, dim=1)
        r0_-=t0
        if laserOrigin!=None:
            r0_+=torch.norm(laserPos-laserOrigin, p=2, dim=1)

        # 计算片元中心的深度
        a = torch.norm(self.gaussians.means-laserPos, p=2, dim=1).unsqueeze(1) # (N,1)
        b = torch.norm(self.gaussians.means-cameraPos, p=2, dim=1).unsqueeze(1) # (N,1)
        r0=a+b+r0_ # (N,1)

        r_=bin_resolution*torch.arange(1,1+num_bins,dtype=torch.float32).to(self.device).flatten() # (M,)
        r=r_.view(1,num_bins) #(1,M)

        sigma=torch.mean(self.gaussians.get_scaling,dim=1).unsqueeze(1) # (N,1)
        sigma=torch.clip(sigma,bin_resolution/2) #一定要不小于分辨率才能保证数值稳定

        # 概率密度,[N,M]
        pdf1=math.sqrt(1/math.pi)*torch.exp(-((r-r0)**2/4/sigma**2))/2/sigma # 高斯分布
        pdf2=(r-r0)*torch.exp(-((r-r0)**2/4/sigma**2))/(2*sigma**2) # 瑞利分布
        pdf=coeff*pdf1+(1-coeff)*pdf2 # 两个分布叠加
        pr=pdf*bin_resolution
        # print(torch.sum(pr,dim=1))
        pr=torch.clip(pr,0,1)

        hist=intensity.unsqueeze(1)/((a*b)) # (N,1)
        hist=hist*pr # (N,M)
        hist=torch.sum(hist,dim=0).flatten()

        # hist[0:100]=0

        return hist