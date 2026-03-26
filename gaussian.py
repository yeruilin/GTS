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

class Gaussians:

    def __init__(
        self, init_type: str, device: str, load_path: Optional[str] = None,
        num_points: Optional[int] = None, isotropic: Optional[bool] = None,
        colour_dim=3,extent=1.0,center=(0,0,0),scale=0.01,view_num=1,means=None,
        use_sigmoid=True # 需要体渲染用平方，不需要用sigmoid
    ):

        self.device = device
        self.colour_dim=colour_dim
        self.view_num=view_num

        # activation function
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.coeff_activate = torch.sigmoid
        self.inverse_coeff_activate = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        self.opacity_activation = torch.sigmoid
        if use_sigmoid==False:
            self.opacity_activation = lambda x: x**2

        if init_type == "gaussians":
            data, is_isotropic = self._load_gaussians(load_path)
            self.is_isotropic = is_isotropic
            self.rotations=data["rotations"]

        elif init_type == "points":
            data = self._load_points(means,scale)

        elif init_type == "random":
            if isotropic is None:
                self.is_isotropic = False
            else:
                self.is_isotropic = isotropic

            data = self._load_random(num_points,extent,center,scale)

        else:
            raise ValueError(f"Invalid init_type: {init_type}")

        # 优化变量
        self.means = data["means"]
        self.scales = data["scales"]
        self.colours = data["colours"]
        self.opacities = data["opacities"]
        self.coefficients = data["coefficients"]
        self.sphere=False

        if self.device!="cpu":
            self.to(self.device)

    def __len__(self):
        return len(self.means)

    def _load_gaussians(self, ply_path: str):

        data = dict()
        ply_gaussians = load_gaussians_from_ply(ply_path)

        data["means"] = torch.tensor(ply_gaussians["xyz"])
        data["rotations"] = torch.tensor(ply_gaussians["rot"])
        data["scales"] = torch.tensor(ply_gaussians["scale"])
        data["opacities"] = torch.tensor(ply_gaussians["opacity"]).squeeze()
        data["colours"] = torch.tensor(ply_gaussians["dc_colours"])
        data["coefficients"] = torch.zeros((len(data["means"]),1), dtype=torch.float32)  # (N,)

        if self.colour_dim==1:
            data["colours"]=data["colours"][:,0:1]

        is_isotropic = False
        if data["scales"].shape[1] != 3:
            is_isotropic=True

        # 计算场景大致范围
        self.center=torch.mean(data["means"],dim=0)
        _range=torch.max(data["means"],dim=0)[0]-torch.min(data["means"],dim=0)[0]
        self.radius=torch.max(_range/2.0).to(self.device)

        return data, is_isotropic

    def _load_points(self, means: np.ndarray,scale=0.01):

        data = dict()
        
        num_points=means.shape[0]

        # Initializing means using the provided point cloud
        data["means"] = torch.tensor(means.astype(np.float32))  # (N, 3)

        # Initializing opacities such that all when sigmoid is applied to opacities
        data["opacities"] = 0.1*torch.ones((num_points,self.view_num), dtype=torch.float32)  # (N,)

        data["coefficients"] = torch.zeros((num_points,1), dtype=torch.float32)  # (N,)

        # Initializing colors randomly
        data["colours"] = 0.01*torch.ones((num_points, self.colour_dim), dtype=torch.float32)  # (N, colour_dim)

        # Initializing scales
        data["scales"] = torch.log(torch.ones((num_points,1),dtype=torch.float32)*scale)  # (N, 1)
        
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
        elif type(radius)==type(np.array([])):
            self.radius=torch.Tensor(radius)
        else:
            self.radius=radius
            
        data["means"]= self.radius*(means_-0.5)*2
        data["means"]+=self.center.view(1,3)

        # Initializing opacities such that all when sigmoid is applied to opacities
        data["opacities"] = 0.1*torch.ones((num_points,self.view_num), dtype=torch.float32)  # (N,)

        data["coefficients"] = torch.zeros((num_points,1), dtype=torch.float32)  # (N,)

        # Initializing colors randomly
        data["colours"] = 0.01*torch.ones((num_points, self.colour_dim), dtype=torch.float32)  # (N, colour_dim)

        # Initializing scales
        data["scales"] = torch.log(torch.ones((num_points,1),dtype=torch.float32)*scale)  # (N, 1)

        if not self.is_isotropic:
            data["scales"] = data["scales"].repeat(1, 3)  # (N, 3)

        return data

    def check_if_trainable(self):

        attrs = ["means", "scales", "colours", "opacities"]
        if not self.is_isotropic:
            attrs += ["rotations"]

        for attr in attrs:
            param = getattr(self, attr)
            if not getattr(param, "requires_grad", False):
                raise Exception("Please use function make_trainable to make parameters trainable")

        if self.is_isotropic and self.rotations.requires_grad:
            raise RuntimeError("You do not need to optimize quaternions in isotropic mode.")

    def to(self,device):
        self.device=device
        self.means = self.means.to(self.device)
        self.scales = self.scales.to(self.device)
        self.colours = self.colours.to(self.device)
        self.opacities = self.opacities.to(self.device)
        self.coefficients = self.coefficients.to(self.device)

        # [Q 1.3.1] NOTE: Uncomment spherical harmonics code for question 1.3.1
        if self.sphere:
            self.spherical_harmonics = self.spherical_harmonics.to(self.device)
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self.scales)
    @property
    def get_opacity(self):
        return self.opacity_activation(self.opacities) # 激活函数对体渲染结果影响很大
    @property
    def get_coefficient(self):
        return self.coeff_activate(self.coefficients)
    @property
    def get_xyz(self):
        return self.means
    @property
    def get_colour(self):
        return self.colours**2
    
    # Inria风格的clone
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
        new_opacities = self.opacities.repeat(copy_num,1)
        new_coefficients = self.coefficients.repeat(copy_num,1)

        self.densification_postfix(new_xyz, new_colours, new_opacities, new_scaling,new_coefficients)
    
    # 四叉树风格的clone
    def densify_and_clone2(self,selected_pts_mask,step,copy_num=4):
        # 用于替代之前的片元
        new_xyz_=self.get_xyz[selected_pts_mask]
        new_scaling_ = self.scales[selected_pts_mask]  # 拷贝后scale下降
        new_colours_ = self.colours[selected_pts_mask]
        new_opacity_ = self.opacities[selected_pts_mask]
        new_coefficients_=self.coefficients[selected_pts_mask]

        # 拷贝后片元中心略有平移
        new_xyz = new_xyz_[None,:,:].repeat(copy_num, 1, 1)
        new_xyz[0, :, 0:2] -= step/2  # 第一份: x - step, y - step
        new_xyz[1, :, 0] += step/2  # 第二份: x + step, y - step
        new_xyz[1, :, 1] -= step/2
        new_xyz[2, :, 1] += step/2  # 第三份: x - step, y + step
        new_xyz[2, :, 0] -= step/2
        new_xyz[3, :, 0:2] += step/2  # 第四份: x + step, y + step
        new_xyz = new_xyz.view(-1, 3)

        new_scaling=new_scaling_.repeat(copy_num,1)
        # 拷贝后旋转、颜色、透明度不变
        new_colours = new_colours_.repeat(copy_num,1)
        new_opacity=new_opacity_.repeat(copy_num,1)
        new_coefficients=new_coefficients_.repeat(copy_num,1)
        
        print(f"clone number: {torch.sum(selected_pts_mask).item()*copy_num}")

        # 将新的split产生的tensor和之前的tensor合并到一起
        self.densification_postfix(new_xyz, new_colours, new_opacity, new_scaling,new_coefficients)
        # 再把分裂前的tensor删除掉
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(copy_num * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(prune_filter)

    # Inria风格的densify
    def density_and_split1(self,selected_pts_mask,copy_num=2,save_old=True):
        if torch.sum(selected_pts_mask)==0:
            print("no points to split")
            return
        stds = self.get_scaling[selected_pts_mask].repeat(copy_num,1)
        means =torch.zeros((stds.size(0), 3),device=self.device)
        samples = torch.normal(mean=means, std=stds)
        
        # 用于替代之前的片元
        new_xyz_=self.get_xyz[selected_pts_mask]
        new_scaling_ = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask] / (0.7*copy_num)) # 拷贝后scale下降
        new_colours_ = self.colours[selected_pts_mask]
        new_opacity_ = self.opacities[selected_pts_mask]
        new_coefficients_=self.coefficients[selected_pts_mask]

        # 拷贝后片元中心略有平移
        new_xyz = samples + new_xyz_.repeat(copy_num, 1)
        new_scaling=new_scaling_.repeat(copy_num,1)
        # 拷贝后旋转、颜色、透明度不变
        new_colours = new_colours_.repeat(copy_num,1)
        new_opacity=new_opacity_.repeat(copy_num,1)
        new_coefficients=new_coefficients_.repeat(copy_num,1)

        # 将新的split产生的tensor和之前的tensor合并到一起
        self.densification_postfix(new_xyz, new_colours, new_opacity, new_scaling,new_coefficients)
        # 再把分裂前的tensor删除掉
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(copy_num * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(prune_filter)
        
        # 之前的位置依然保留，但是大小下降
        if save_old:
            self.densification_postfix(new_xyz_, new_colours_, new_opacity_, new_scaling_,new_coefficients_)

    
    def set_scale(self,selected_pts_mask,new_scale):
        # 用于替代之前的片元
        new_xyz_=self.get_xyz[selected_pts_mask]
        new_scaling_ = self.scaling_inverse_activation(torch.ones_like(self.get_scaling[selected_pts_mask],device=self.device)*new_scale) # 拷贝后scale下降
        new_colours_ = self.colours[selected_pts_mask]
        new_opacity_ = self.opacities[selected_pts_mask]
        new_coefficients_ = self.coefficients[selected_pts_mask]

        # 再把分裂前的tensor删除掉
        self.prune_points(selected_pts_mask)
        # 之前的位置依然保留，但是大小下降
        self.densification_postfix(new_xyz_, new_colours_, new_opacity_, new_scaling_,new_coefficients_)

    # def training_setup(self, training_args):
    #     self.scales.requires_grad=True
    #     self.colours.requires_grad=True
    #     self.opacities.requires_grad=True

    #     l = [
    #         # {'params': [self.means], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
    #         {'params': [self.colours], 'lr': training_args.feature_lr, "name": "colours"},
    #         {'params': [self.opacities], 'lr': training_args.opacity_lr, "name": "opacity"},
    #         {'params': [self.scales], 'lr': training_args.scaling_lr, "name": "scaling"}
    #     ]

    #     self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def training_setup(self,l_dict=None,train_fast=True):
        self.scales.requires_grad=True
        self.colours.requires_grad=True
        self.opacities.requires_grad=True
        self.coefficients.requires_grad=True

        if l_dict==None:
            if train_fast:
                l = [
                    {'params': [self.colours], 'lr': 0.0025, "name": "colour"},
                    {'params': [self.coefficients], 'lr': 0.02, "name": "coefficient"},
                    {'params': [self.opacities], 'lr': 0.02, "name": "opacity"},
                    {'params': [self.scales], 'lr': 0.002, "name": "scaling"}
                ]
            else:
                l = [
                        {'params': [self.colours], 'lr': 0.001, "name": "colour"},
                        {'params': [self.coefficients], 'lr': 0.01, "name": "coefficient"},
                        {'params': [self.opacities], 'lr': 0.01, "name": "opacity"},
                        {'params': [self.scales], 'lr': 0.001, "name": "scaling"}
                    ]
        else:
            l=l_dict

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.8)
    
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
        
        if torch.sum(valid_points_mask)==0:
            print("no points to prune")
            return
        else:
            print(f"prune number: {torch.sum(mask).item()}")

        self.means = self.means[valid_points_mask]

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self.colours = optimizable_tensors["colour"]
        self.opacities = optimizable_tensors["opacity"]
        self.scales = optimizable_tensors["scaling"]
        self.coefficients = optimizable_tensors["coefficient"]

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

    def densification_postfix(self, new_xyz, new_colours, new_opacities, new_scaling,new_coefficient):
        self.means = torch.cat([self.means,new_xyz],dim=0)

        d = {
        "colour": new_colours,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "coefficient" : new_coefficient
        }

        # 将新生成的tensor拼接到之前的变量上
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        # 然后重新赋值
        self.colours = optimizable_tensors["colour"]
        self.opacities = optimizable_tensors["opacity"]
        self.scales = optimizable_tensors["scaling"]
        self.coefficients = optimizable_tensors["coefficient"]
    
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

    # 利用极坐标的形式计算histogram
    def render_conf_hist(self, scan_point,bin_resolution,num_bins,t0,decay=2.0,view_id=0):
        # 计算强度
        if type(view_id)!=type(0):
            view_id=view_id.item()

        # 计算强度
        intensity=self.get_opacity[:,view_id].flatten()*self.get_colour.flatten() # (N,)

        # 计算片元中心的深度:scan_point is [1,3]
        r0 = torch.norm(self.means-scan_point, p=2, dim=1).unsqueeze(1) # (N,1)

        r_=t0/2+bin_resolution/2*torch.arange(1,1+num_bins,dtype=torch.float32).to(self.device).flatten() # (M,)
        r=r_.view(1,num_bins) #(1,M)

        sigma=torch.mean(self.get_scaling,dim=1).unsqueeze(1) # (N,1)
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
    def render_conf_hist2(self, scan_point,bin_resolution,num_bins,t0=0,decay=2.0,view_id=0):
        # 计算强度
        if type(view_id)!=type(0):
            view_id=view_id.item()
            
        intensity=self.get_opacity[:,view_id].flatten()*self.get_colour.flatten() # (N,)

        # 计算两组基的系数
        coeff=self.get_coefficient.flatten().unsqueeze(1) # (N,1)

        # 计算片元中心的深度:scan_point is [1,3]
        r0 = torch.norm(self.means-scan_point, p=2, dim=1).unsqueeze(1) # (N,1)

        r_=t0/2+bin_resolution/2*torch.arange(1,1+num_bins,dtype=torch.float32).to(self.device).flatten() # (M,)
        r=r_.view(1,num_bins) #(1,M)

        sigma=torch.mean(self.get_scaling,dim=1).unsqueeze(1) # (N,1)
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
    def render_nonconf_hist(self, laserPos,laserOrigin,cameraPos,cameraOrigin,bin_resolution,num_bins,t0=0,view_id=0):
        # 计算强度
        if type(view_id)!=type(0):
            view_id=view_id.item()

        # 计算强度
        intensity=self.get_opacity[:,view_id].flatten()*self.get_colour.flatten() # (N,)

        # 计算激光点和相机点到两边的距离
        r0_=torch.norm(cameraPos-cameraOrigin, p=2, dim=1)
        r0_-=t0
        if laserOrigin!=None:
            r0_+=torch.norm(laserPos-laserOrigin, p=2, dim=1)

        # 计算片元中心的深度
        a = torch.norm(self.means-laserPos, p=2, dim=1).unsqueeze(1) # (N,1)
        b = torch.norm(self.means-cameraPos, p=2, dim=1).unsqueeze(1) # (N,1)
        r0=a+b+r0_ # (N,1)

        r_=bin_resolution*torch.arange(1,1+num_bins,dtype=torch.float32).to(self.device).flatten() # (M,)
        r=r_.view(1,num_bins) #(1,M)

        sigma=torch.mean(self.get_scaling,dim=1).unsqueeze(1) # (N,1)
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

    def render_nonconf_hist2(self, laserPos,laserOrigin,cameraPos,cameraOrigin,bin_resolution,num_bins,t0=0,view_id=0):
        # 计算强度
        if type(view_id)!=type(0):
            view_id=view_id.item()
            
        intensity=self.get_opacity[:,view_id].flatten()*self.get_colour.flatten() # (N,)

        # 计算两组基的系数
        coeff=self.get_coefficient.flatten().unsqueeze(1) # (N,1)

        # 计算激光点和相机点到两边的距离
        r0_=torch.norm(cameraPos-cameraOrigin, p=2, dim=1)
        r0_-=t0
        if laserOrigin!=None:
            r0_+=torch.norm(laserPos-laserOrigin, p=2, dim=1)

        # 计算片元中心的深度
        a = torch.norm(self.means-laserPos, p=2, dim=1).unsqueeze(1) # (N,1)
        b = torch.norm(self.means-cameraPos, p=2, dim=1).unsqueeze(1) # (N,1)
        r0=a+b+r0_ # (N,1)

        r_=bin_resolution*torch.arange(1,1+num_bins,dtype=torch.float32).to(self.device).flatten() # (M,)
        r=r_.view(1,num_bins) #(1,M)

        sigma=torch.mean(self.get_scaling,dim=1).unsqueeze(1) # (N,1)
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