import torch
import torch.distributed as dist
import torch.nn as nn

import numpy as np
import os
import math
from dataset import *

import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

class GaussianModel(nn.Module):
    def __init__(self, num_points, scale,
                 bin_resolution=0.01,num_bins=512,t0=0,decay=2.0,confocal=True,volume_render=False,
                 laserOrigin=None,cameraPos=[0,0,0],cameraOrigin=[0,0,0], # nonconfocal
                 view_num=1
                 ):
        super().__init__()

        self.confocal=confocal
        self.num_points=num_points
        self.view_num=view_num
        self.volume_render=volume_render

        # activation function
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        # Initial parameters
        self.colours=nn.Parameter(0.01*torch.ones((self.num_points,1), dtype=torch.float32))
        self.coefficients = nn.Parameter(0.0 * torch.ones((self.num_points,1), dtype=torch.float32))
        self.opacities = nn.Parameter(0.1 * torch.ones((self.num_points,self.view_num), dtype=torch.float32))
        self.scales = nn.Parameter(self.scaling_inverse_activation(torch.ones_like(self.coefficients)*scale))

        # Used by both confocal and nonconfocal
        self.bin_resolution=bin_resolution
        self.num_bins=num_bins
        self.t0=t0
        self.decay=decay

        # Used in nonconfocal
        self.laserOrigin=laserOrigin
        self.cameraPos=cameraPos
        self.cameraOrigin=cameraOrigin

    def parameters(self):
        return [self.scales, self.colours, self.coefficients,self.opacities]
    
    def get_all_parameters(self):
        return {
            'scale': self.get_scaling.clone().detach(),
            'rho': self.get_colour.clone().detach(),
            'c': self.get_coefficient.clone().detach(),
            'o': self.get_opacity.clone().detach()
        }
    
    def __len__(self):
        return self.num_points
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self.scales)
    @property
    def get_coefficient(self):
        return self.opacity_activation(self.coefficients)
    @property
    def get_opacity(self):
        return self.opacities**2
    @property
    def get_colour(self):
        return self.colours**2

    def _compute_jacobian(self, means_3D: torch.Tensor, camera, img_size:tuple):

        # FoVCamera
        W, H = img_size
        half_tan_fov_x=torch.tan(camera.fov)/2
        half_tan_fov_y=torch.tan(camera.fov)/2
        fx=W/torch.tan(camera.fov)
        fy=H/torch.tan(camera.fov)

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
        J = J.to(means_3D.device)

        J[:, 0, 0] = fx / tz
        J[:, 1, 1] = fy / tz
        J[:, 0, 2] = -(fx * tx) / tz2
        J[:, 1, 2] = -(fy * ty) / tz2

        return J  # (N, 2, 3)
    
    def compute_cov_2D(self, means_3D: torch.Tensor, scales: torch.Tensor, camera, img_size: tuple):

        N=scales.shape[0]
        J = self._compute_jacobian(means_3D,camera,img_size)  # (N, 2, 3)
        W = camera.R.repeat((N,1,1))  # (N, 3, 3)

        # calculate cov 3D
        scales=scales*scales
        scales=scales.repeat((1,3))
        cov_3D=torch.diag_embed(scales) # (N, 3, 3)

        J=torch.bmm(J,W)
        cov_2D = torch.bmm(J,torch.bmm(cov_3D,J.transpose(1,2)))  # (N, 2, 2)

        # Post processing to make sure that each 2D Gaussian covers atleast approximately 1 pixel
        cov_2D[:, 0, 0] += 0.3
        cov_2D[:, 1, 1] += 0.3

        return cov_2D
    
    @staticmethod
    def invert_cov_2D(cov_2D: torch.Tensor):
        determinants = cov_2D[:, 0, 0] * cov_2D[:, 1, 1] - cov_2D[:, 1, 0] * cov_2D[:, 0, 1]
        determinants = determinants[:, None, None]  # (N, 1, 1)
        cov_2D_inverse = torch.zeros_like(cov_2D)  # (N, 2, 2)
        cov_2D_inverse[:, 0, 0] = cov_2D[:, 1, 1]
        cov_2D_inverse[:, 1, 1] = cov_2D[:, 0, 0]
        cov_2D_inverse[:, 0, 1] = -1.0 * cov_2D[:, 0, 1]
        cov_2D_inverse[:, 1, 0] = -1.0 * cov_2D[:, 1, 0]
        cov_2D_inverse = (1.0 / determinants) * cov_2D_inverse
        return cov_2D_inverse
    
    def compute_alphas(self, opacities, means_2D, cov_2D, img_size):
        W, H = img_size

        # point_2D contains all possible pixel locations in an image
        xs, ys = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
        points_2D = torch.stack((xs.flatten(), ys.flatten()), dim = 1)  # (H*W, 2)
        points_2D = points_2D.to(opacities.device).unsqueeze(0) # (1, H*W, 2)
        means_2D = means_2D.unsqueeze(1)  # (N, 1, 2)

        cov_2D_inverse = self.invert_cov_2D(cov_2D)  # (N, 2, 2) 

        # evaluate_gaussian_2D
        vec=points_2D-means_2D # (N,H*W,2)
        vec2=torch.bmm(cov_2D_inverse,vec.transpose(1,2)).transpose(1,2) # (N,H*W,2)
        power =-0.5*(vec[:,:,0]*vec2[:,:,0]+vec[:,:,1]*vec2[:,:,1])  # (N, H*W)

        # Computing exp(power) with some post processing for numerical stability
        exp_power = torch.where(power > 0.0, 0.0, torch.exp(power))
        alphas = torch.reshape(opacities*exp_power, (-1, H, W))  # (N, H, W)

        # Post processing for numerical stability
        alphas = torch.minimum(alphas, torch.full_like(alphas, 0.99))
        alphas = torch.where(alphas < 1/255.0, 0.0, alphas)
        return alphas

    # 体渲染模式
    def render_conf_hist(self, means, scan_point, camera, img_size=(64,64)):
        # Globally sort gaussians according to their depth value
        z_vals_origin = torch.norm(means-scan_point, p=2, dim=1) # (N,)

        # sort
        idxs = torch.argsort(z_vals_origin)  # (M,)
        idxs = idxs[torch.sum(z_vals_origin<=0):]

        scales = self.get_scaling[idxs] # [N,1]
        z_vals = z_vals_origin[idxs] # [N,1]
        means_3D = means[idxs] # [N,3]
        colours = self.get_colour[idxs] # [N,1]
        coeff=self.get_coefficient[idxs] # [N,1]
        opacities = self.get_opacity[idxs]+0.0*coeff # [N,1]

        ## Volume rendering histogram
        N=means_3D.shape[0]

        # Step 1: Compute 2D gaussian parameters
        means_2D = camera.transform_points_screen(means_3D)[:,:2] # (N, 2)
        cov_2D = self.compute_cov_2D(means_3D,scales,camera,img_size)  # (N, 2, 2)

        # Step 2: Compute alpha maps for each gaussian
        alphas = self.compute_alphas(opacities,means_2D,cov_2D,img_size)  # (N, H, W)

        # Step 3: Compute transmittance maps for each gaussian
        one_minus_alphas = torch.concat((torch.ones((1, img_size[0], img_size[1]), device=alphas.device, dtype=alphas.dtype), 1.0 - alphas), dim=0)  # (N+1, H, W)
        transmittance = torch.cumprod(one_minus_alphas,dim=0)[:-1,:,:]  # (N, H, W)
        transmittance = torch.where(transmittance < 1e-4, 0.0, transmittance)  # (N, H, W)

        # integrate on phi and theta
        intensity=colours[:,:,None]*alphas*transmittance  # (N,H, W)
        intensity = torch.sum(intensity.view(N, -1), dim=1)  # (N,)

        # ## NLOS rendering
        r_=self.t0/2+self.bin_resolution/2*torch.arange(1,1+self.num_bins,dtype=torch.float32).to(means.device).flatten() # (M,)
        r=r_.view(1,self.num_bins) #(1,M)

        sigma=torch.mean(scales,dim=1).unsqueeze(1) # (N,1)
        sigma=torch.clip(sigma,self.bin_resolution/2)

        # pdf,[N,M]
        r0=z_vals.unsqueeze(1)
        pdf=math.sqrt(0.5/math.pi)*torch.exp(-0.5*((r-r0)/sigma)**2)/sigma
        pr=pdf*self.bin_resolution/2 # probability, [N,M]
        pr=torch.clip(pr,0,1)

        hist=intensity.unsqueeze(1)*pr # (N,M)
        hist=torch.sum(hist,dim=0).flatten()
        hist=hist/torch.pow(r_,self.decay)

        return hist
    
    # 利用极坐标的形式计算histogram
    def render_conf_hist2(self,means, scan_point,view_id=0):
        # 计算强度
        if type(view_id)!=type(0):
            view_id=view_id.item()
            
        intensity=self.get_opacity[:,view_id].flatten()*self.get_colour.flatten() # (N,)

        # 计算两组基的系数
        coeff=self.get_coefficient.flatten().unsqueeze(1) # (N,1)

        # 计算片元中心的深度:scan_point is [1,3]
        r0 = torch.norm(means-scan_point, p=2, dim=1).unsqueeze(1) # (N,1)

        r_=self.t0/2+self.bin_resolution/2*torch.arange(1,1+self.num_bins,dtype=torch.float32).to(scan_point.device).flatten() # (M,)
        r=r_.view(1,self.num_bins) #(1,M)

        sigma=torch.mean(self.get_scaling,dim=1).unsqueeze(1) # (N,1)
        sigma=torch.clip(sigma,self.bin_resolution/2) #一定要不小于分辨率才能保证数值稳定

        # 概率密度,[N,M]
        pdf1=math.sqrt(0.5/math.pi)*torch.exp(-0.5*((r-r0)/sigma)**2)/sigma # 高斯分布
        pdf2=(r-r0)*torch.exp(-0.5*((r-r0)/sigma)**2)/(sigma**2) # 瑞利分布
        pdf=coeff*pdf1+(1-coeff)*pdf2 # 两个分布叠加
        pr=pdf*self.bin_resolution/2 # 概率, [N,M]
        pr=torch.clip(pr,0,1)

        # print(torch.mean(torch.sum(pr,1)))

        hist=intensity.unsqueeze(1)*pr # (N,M)
        hist=torch.sum(hist,dim=0).flatten()
        hist=hist/torch.pow(r_,self.decay)

        return hist

    def render_nonconf_hist2(self, means,laserPos,view_id=0):
        # 计算强度
        if type(view_id)!=type(0):
            view_id=view_id.item()
        intensity=self.get_opacity[:,view_id].flatten()*self.get_colour.flatten() # (N,)

        # 计算两组基的系数
        coeff=self.coefficients.flatten().unsqueeze(1) # (N,1)

        # 计算激光点和相机点到两边的距离
        r0_=torch.norm(self.cameraPos-self.cameraOrigin, p=2, dim=1)
        r0_-=self.t0
        if self.laserOrigin!=None:
            r0_+=torch.norm(laserPos-self.laserOrigin, p=2, dim=1)

        # 计算片元中心的深度
        a = torch.norm(means-laserPos, p=2, dim=1).unsqueeze(1) # (N,1)
        b = torch.norm(means-self.cameraPos, p=2, dim=1).unsqueeze(1) # (N,1)
        r0=a+b+r0_ # (N,1)

        r_=self.bin_resolution*torch.arange(1,1+self.num_bins,dtype=torch.float32).to(laserPos.device).flatten() # (M,)
        r=r_.view(1,self.num_bins) #(1,M)

        sigma=torch.mean(self.get_scaling,dim=1).unsqueeze(1) # (N,1)
        sigma=torch.clip(sigma,self.bin_resolution/2) #一定要不小于分辨率才能保证数值稳定

        # 概率密度,[N,M]
        pdf1=math.sqrt(1/math.pi)*torch.exp(-((r-r0)**2/4/sigma**2))/2/sigma # 高斯分布
        pdf2=(r-r0)*torch.exp(-((r-r0)**2/4/sigma**2))/(2*sigma**2) # 瑞利分布
        pdf=coeff*pdf1+(1-coeff)*pdf2 # 两个分布叠加
        pr=pdf*self.bin_resolution
        # print(torch.sum(pr,dim=1))
        pr=torch.clip(pr,0,1)

        hist=intensity.unsqueeze(1)/((torch.pow(a,self.decay)*(torch.pow(b,self.decay)))) # (N,1)
        hist=hist*pr # (N,M)
        hist=torch.sum(hist,dim=0).flatten()

        return hist

    def forward(self,means,scan_point,view_id=0,camera=None):
        if self.volume_render:
            if self.confocal:
                return self.render_conf_hist(means,scan_point,camera)
        else:
            if self.confocal:
                return self.render_conf_hist2(means,scan_point,view_id)
            else:
                return self.render_nonconf_hist2(means,scan_point,view_id)

# def gather_all_parameters(rank, world_size, local_params,pixels):
#     """
#     收集所有GPU上的参数到主GPU
#     返回: 主GPU上包含所有参数的字典
#     """
#     # 为每个参数创建存储列表
#     scale_list = [torch.zeros_like(local_params['scale']) for _ in range(world_size)]
#     rho_list = [torch.zeros_like(local_params['rho']) for _ in range(world_size)]
#     o_list = [torch.zeros_like(local_params['c']) for _ in range(world_size)]
    
#     # 收集所有GPU的参数
#     dist.all_gather(scale_list, local_params['scale'])
#     dist.all_gather(rho_list, local_params['rho'])
#     dist.all_gather(o_list, local_params['c'])
    
#     if rank == 0:
#         # 在主GPU上拼接所有参数
#         rho = torch.zeros(pixels[0], pixels[1], pixels[2], dtype=torch.float32, device="cpu")
#         rho[0::2, 0::2,:] = rho_list[0].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()# 按照奇偶位置插入
#         rho[0::2, 1::2,:] = rho_list[1].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()
#         rho[1::2, 0::2,:] = rho_list[2].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()
#         rho[1::2, 1::2,:] = rho_list[3].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()

#         scale = torch.zeros(pixels[0], pixels[1], pixels[2], dtype=torch.float32, device="cpu")
#         scale[0::2, 0::2,:] = scale_list[0].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()
#         scale[0::2, 1::2,:] = scale_list[1].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()
#         scale[1::2, 0::2,:] = scale_list[2].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()
#         scale[1::2, 1::2,:] = scale_list[3].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()

#         o = torch.zeros(pixels[0], pixels[1], pixels[2], dtype=torch.float32, device="cpu")
#         o[0::2, 0::2,:] = o_list[0].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()
#         o[0::2, 1::2,:] = o_list[1].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()
#         o[1::2, 0::2,:] = o_list[2].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()
#         o[1::2, 1::2,:] = o_list[3].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu()
        
#         return {"rho":rho.numpy(),"scale":scale.numpy(),"o":o.numpy()}
    
#     # if rank == 0:
#     #     # 在主GPU上拼接所有参数
#     #     rho = local_params['rho'].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu().numpy()
#     #     scale = local_params['scale'].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu().numpy()
#     #     o = local_params['o'].view(pixels[0]//2,pixels[1]//2,pixels[2]).cpu().numpy()
#     #     return {"rho":rho,"scale":scale,"o":o}
    
#     return None

def save_parameters(all_params, save_dir="temp"):
    os.makedirs(save_dir, exist_ok=True)

    numpy_params = {k: v.numpy() for k, v in all_params.items()}
    torch.save(numpy_params, os.path.join(save_dir, "result.npz"))
    
    print(f"All parameters saved to {save_dir}")


def makegrid(minimalpos, maximalpos, gridsize,rank,world_size):
    assert world_size==4
    xx=rank//2
    yy=rank-2*xx

    minimalpos = np.asarray(minimalpos)
    maximalpos = np.asarray(maximalpos)
    gridsize = np.asarray(gridsize)

    # Number of pixels per direction
    pixels = np.ceil(np.abs(minimalpos - maximalpos) /2/gridsize).astype(int)*2

    # Unit vectors scaled by grid size
    vx = np.array([1, 0, 0]) * gridsize
    vy = np.array([0, 1, 0]) * gridsize
    vz = np.array([0, 0, 1]) * gridsize

    # Generate grid points
    pts = []
    for x in range(xx,pixels[0],2):
        for y in range(yy,pixels[1],2):
            for z in range(pixels[2]):
                tmp = minimalpos + x * vx + y * vy + z * vz
                pts.append(tmp)

    pts = np.array(pts)
    return pts,pixels