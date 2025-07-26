import math
import torch
import numpy as np

from typing import Tuple, Optional
from gaussian import Gaussians
from pytorch3d.renderer.cameras import PerspectiveCameras,FoVPerspectiveCameras
from pytorch3d.transforms import quaternion_to_matrix

from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings

class Scene:
    def __init__(self, gaussians: Gaussians):
        self.gaussians = gaussians
        self.device = self.gaussians.device
    
    def render(self, mycamera, per_splat: int = -1,img_size: Tuple = (256, 256), bg_colour: Tuple = (0.0, 0.0, 0.0)):
        """
        Render the scene. 目前只能支持在cuda:0运行
        """
        image_height, image_width = img_size

        bg_color=torch.tensor([0,0,0], device=self.device).float()

        # 相机参数
        view = mycamera.get_world_to_view_transform().get_matrix().contiguous().to(self.device) # (4, 4) camera extrinsics
        perspective=mycamera.get_full_projection_transform().get_matrix().contiguous().to(self.device) # (4,4) camera intrinsics
        camera_center = torch.inverse(view)[:3, 3]
        
        means3D=self.gaussians.means # [N,3]
        
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=self.device) + 0 # [N,3]
        try:
            screenspace_points.retain_grad()
        except:
            pass
        # Set up rasterization configuration
        if type(mycamera)==PerspectiveCameras:
            fx, fy = mycamera.focal_length.flatten()
            tanfovx = (image_height / fx).item()
            tanfovy = (image_width / fy).item()
        elif type(mycamera)==FoVPerspectiveCameras:
            tanfovx = torch.tan(mycamera.fov).item()
            tanfovy = torch.tan(mycamera.fov).item()
        
        subpixel_offset = torch.zeros((image_height, image_width, 2), dtype=torch.float32, device=self.device)

        raster_settings = GaussianRasterizationSettings(
            image_height=image_height,
            image_width=image_width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            kernel_size=0.1, # 我也不知道什么意思
            subpixel_offset=subpixel_offset,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=view,
            projmatrix=perspective,
            sh_degree=0, # 只有直流分量
            campos=camera_center,
            prefiltered=False,
            debug=True
        )
        
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = screenspace_points # [N,3]
        opacity = self.gaussians.get_opacity # [N,1]

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.

        scales=self.gaussians.get_scaling.repeat((1,3))
        cov3D_precomp=torch.diag_embed(scales)
        scales=None # [N,3]
        rotations=None # [N,4]
        # rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]],dtype=torch.float32).repeat(means3D.shape[0], 1).to(self.device)  # Identity rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        colors_precomp = self.gaussians.get_colour.repeat(1,3) # [N,3]
        shs = None # [N,1,3]

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp
        )
        # rendered_image is torch.float32, shape is [3,image_height, image_width]
        rendered_image = rendered_image.permute(1, 2, 0)  # Change to [image_height, image_width, 3]
        return rendered_image,rendered_image[:,:,0:1],rendered_image[:,:,0:1] # depth and mask are not used in this case

    # def render(
    #     self, camera: PerspectiveCameras,
    #     per_splat: int = -1, img_size: Tuple = (256, 256),
    #     bg_colour: Tuple = (0.0, 0.0, 0.0),
    #     no_grad=False
    # ):
    #     """
    #     Given a scene represented by N 3D Gaussians, this function renders the RGB
    #     colour image, the depth map and the silhouette map that can be observed
    #     from a given pytorch 3D camera.

    #     Args:
    #         camera      :   A pytorch3d PerspectiveCameras object.
    #         per_splat   :   Number of gaussians to splat in one function call. If set to -1,
    #                         then all gaussians in the scene are splat in a single function call.
    #                         If set to any other positive interger, then it determines the number of
    #                         gaussians to splat per function call (the last function call might splat
    #                         lesser number of gaussians). In general, the algorithm can run faster
    #                         if more gaussians are splat per function call, but at the cost of higher GPU
    #                         memory consumption.
    #         img_size    :   The (width, height) of the image to be rendered.
    #         bg_color    :   A tuple indicating the RGB colour that the background should have.

    #     Returns:
    #         image       :   A torch.Tensor of shape (H, W, 3) with the rendered RGB colour image.
    #         depth       :   A torch.Tensor of shape (H, W, 1) with the rendered depth map.
    #         mask        :   A torch.Tensor of shape (H, W, 1) with the rendered silhouette map.
    #     """
    #     bg_colour_ = torch.tensor(bg_colour)[None, None, :].to(self.device)  # (1, 1, 3)

    #     # Globally sort gaussians according to their depth value
    #     z_vals = self.compute_depth_values(camera)
    #     idxs = self.get_idxs_to_filter_and_sort(z_vals)
    #     colours = self.gaussians.get_colour[idxs]
    #     quats = torch.zeros([colours.shape[0],4],dtype=colours.dtype,device=colours.device)
    #     quats[:,3]=1.0
    #     scales = self.gaussians.get_scaling[idxs]
    #     opacities = self.gaussians.get_opacity[idxs]
    #     z_vals = z_vals[idxs]
    #     means_3D = self.gaussians.means[idxs]

    #     if per_splat == -1:
    #         num_mini_batches = 1
    #     elif per_splat > 0:
    #         num_mini_batches = math.ceil(len(means_3D) / per_splat)
    #     else:
    #         raise ValueError("Invalid setting of per_splat")

    #     # In this case we can directly splat all gaussians onto the image
    #     if num_mini_batches == 1:

    #         # Get image, depth and mask via splatting
    #         image, depth, mask,_ = self.splat(
    #             camera, means_3D, z_vals, quats, scales,
    #             colours, opacities, img_size,no_grad=no_grad
    #         )

    #     # In this case we splat per_splat number of gaussians per iteration. This makes
    #     # the implementation more memory efficient but at the same time makes it slower.
    #     else:

    #         W, H = img_size
    #         D = means_3D.device
    #         start_transmittance = torch.ones((1, H, W), dtype=torch.float32).to(D)
    #         image = torch.zeros((H, W, 3), dtype=torch.float32).to(D)
    #         depth = torch.zeros((H, W, 1), dtype=torch.float32).to(D)
    #         mask = torch.zeros((H, W, 1), dtype=torch.float32).to(D)

    #         # 每次计算一部分片元的颜色，最后叠加起来得到总的结果
    #         for b_idx in range(num_mini_batches):

    #             quats_ = quats[b_idx * per_splat: (b_idx+1) * per_splat]
    #             scales_ = scales[b_idx * per_splat: (b_idx+1) * per_splat]
    #             z_vals_ = z_vals[b_idx * per_splat: (b_idx+1) * per_splat]
    #             colours_ = colours[b_idx * per_splat: (b_idx+1) * per_splat]
    #             means_3D_ = means_3D[b_idx * per_splat: (b_idx+1) * per_splat]
    #             opacities_ = opacities[b_idx * per_splat: (b_idx+1) * per_splat]

    #             # Get image, depth and mask via splatting
    #             image_, depth_, mask_, start_transmittance = self.splat(
    #                 camera, means_3D_, z_vals_, quats_, scales_, colours_,
    #                 opacities_, img_size, start_transmittance,no_grad=no_grad
    #             ) # 这里means_2D没有累加，此模式运行有问题，必须一次全导入

    #             image = image + image_
    #             depth = depth + depth_
    #             mask = mask + mask_

    #     # image = mask * image + (1.0 - mask) * bg_colour_
    #     if image.shape[-1]==1:
    #         image=image.repeat(1,1,3)

    #     return image, depth, mask
    
    def nlos_splat(self,means_3D,colours,opacities,scales,quats,camera,img_size,b_idx=None):
        ## Volume rendering histogram
        N=means_3D.shape[0]

        # Step 1: Compute 2D gaussian parameters
        means_2D = self.compute_means_2D(means_3D,camera)  # (N, 2)
        cov_2D = self.compute_cov_2D(means_3D,quats,scales,camera,img_size)  # (N, 2, 2)

        # Step 2: Compute alpha maps for each gaussian
        alphas = self.compute_alphas(opacities,means_2D,cov_2D,img_size,b_idx)  # (N, laser_num)

        # Step 3: Compute transmittance maps for each gaussian
        transmittance = self.compute_transmittance(alphas)  # (N, laser_num)

        # integrate on phi and theta
        intensity=colours*alphas*transmittance  # (N,laser_num)
        intensity=torch.sum(intensity,dim=1)

        return intensity
    
    def render_conf_hist(self,camera,bin_resolution,num_bins,t0=0,decay=4,per_splat=-1,img_size=(64,64)):
        # Globally sort gaussians according to their depth value
        z_vals_origin = self.compute_depth_values(camera) # (N,)
        idxs = self.get_idxs_to_filter_and_sort(z_vals_origin)

        scales = self.gaussians.get_scaling[idxs] # [N,1]
        opacities = self.gaussians.get_opacity[idxs] # [N,1]
        quats= torch.zeros([scales.shape[0],4],dtype=scales.dtype,device=scales.device)
        z_vals = z_vals_origin[idxs] # [N,1]
        means_3D = self.gaussians.means[idxs] # [N,3]
        colours = self.gaussians.get_colour[idxs] # [N,1]

        N=means_3D.shape[0]

        # In this case we can directly splat all gaussians onto the image
        if per_splat == -1:
            intensity=self.nlos_splat(means_3D,colours,opacities,scales,quats,camera,img_size)
        else:
            W, H = img_size
            D = means_3D.device
            intensity = torch.zeros((N,), dtype=torch.float32).to(D)

            num_mini_batches = math.ceil(W/ per_splat)

            # 每次计算一部分片元的颜色，最后叠加起来得到总的结果
            for i in range(num_mini_batches):
                b_idx=[i*per_splat,(i+1)*per_splat]
                intensity_=self.nlos_splat(means_3D,colours,opacities,scales,quats,camera,img_size,b_idx)
                intensity =intensity+ intensity_

        # ## NLOS rendering
        # hist_inten=intensity/(z_vals**decay) # 形状是[select_num]
        # indices=(z_vals*2/bin_resolution).long() # 计算索引
        # indices = torch.clamp(indices, 0, num_bins - 1).flatten()  # 防止索引超出范围

        # hist=torch.zeros((num_bins,),dtype=torch.float32,device=camera.device) # 这里不要写require梯度，因为这个内存要在scatter_add_的时候被占掉
        # hist.scatter_add_(0, indices, hist_inten.flatten())

        # ## NLOS rendering
        r_=t0/2+bin_resolution/2*torch.arange(1,1+num_bins,dtype=torch.float32).to(self.device).flatten() # (M,)
        r=r_.view(1,num_bins) #(1,M)

        sigma=torch.mean(scales,dim=1).unsqueeze(1) # (N,1)
        sigma=torch.clip(sigma,bin_resolution/2) #一定要不小于分辨率才能保证数值稳定

        # 概率密度,[N,M]
        r0=z_vals.unsqueeze(1)
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