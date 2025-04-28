import math
import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, Optional
from pytorch3d.ops.knn import knn_points
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.renderer.cameras import PerspectiveCameras
from data_utils import load_gaussians_from_ply, colours_from_spherical_harmonics

class Scene:
    def __init__(self, gaussians: Gaussians):
        self.gaussians = gaussians
        self.device = self.gaussians.device

    def compute_depth_values(self, camera: PerspectiveCameras):
        means_2D = camera.get_world_to_view_transform().transform_points(self.gaussians.means) # 世界坐标系转相机坐标系
        z_vals = means_2D[:,2]  # (N,)
        return z_vals

    def get_idxs_to_filter_and_sort(self, z_vals: torch.Tensor):

        idxs = torch.argsort(z_vals)  # (M,)

        count = torch.sum(z_vals<=0)
        
        return idxs[count:]

    def compute_alphas(self, opacities, means_2D, cov_2D, img_size):

        W, H = img_size

        # point_2D contains all possible pixel locations in an image
        xs, ys = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
        points_2D = torch.stack((xs.flatten(), ys.flatten()), dim = 1)  # (H*W, 2)
        points_2D = points_2D.to(self.device)

        points_2D = points_2D.unsqueeze(0)  # (1, H*W, 2)
        means_2D = means_2D.unsqueeze(1)  # (N, 1, 2)

        cov_2D_inverse = Gaussians.invert_cov_2D(cov_2D)  # (N, 2, 2) TODO: Verify shape

        power = Gaussians.evaluate_gaussian_2D(points_2D,means_2D,cov_2D_inverse)  # (N, H*W)

        # Computing exp(power) with some post processing for numerical stability
        exp_power = torch.where(power > 0.0, 0.0, torch.exp(power))

        alphas = opacities.unsqueeze(1)*exp_power  # (N, H*W)
        alphas = torch.reshape(alphas, (-1, H, W))  # (N, H, W)

        # Post processing for numerical stability
        alphas = torch.minimum(alphas, torch.full_like(alphas, 0.99))
        alphas = torch.where(alphas < 1/255.0, 0.0, alphas)

        return alphas

    def compute_transmittance(
        self, alphas: torch.Tensor,
        start_transmittance: Optional[torch.Tensor] = None
    ):
        """
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

        transmittance = torch.cumprod(one_minus_alphas,dim=0)  
        transmittance=transmittance[:-1,:,:] # (N, H, W)

        # Post processing for numerical stability
        transmittance = torch.where(transmittance < 1e-4, 0.0, transmittance)  # (N, H, W)

        return transmittance

    def splat(
        self, camera: PerspectiveCameras, means_3D: torch.tensor, z_vals: torch.Tensor,
        quats: torch.Tensor, scales: torch.Tensor, colours: torch.Tensor,
        opacities: torch.Tensor, img_size: Tuple = (256, 256),
        start_transmittance: Optional[torch.Tensor] = None,
        no_grad=False
    ):
        """
        Given N ordered (depth-sorted) 3D Gaussians (or equivalently in our case,
        the parameters of the 3D Gaussians like means, quats etc.), this function splats
        them to the image plane to render an RGB image, depth map and a silhouette map.

        Args:
            camera                  :   A pytorch3d PerspectiveCameras object.
            means_3D                :   A torch.Tensor of shape (N, 3) with the means
                                        of the 3D Gaussians.
            z_vals                  :   A torch.Tensor of shape (N,) with the depths
                                        of the 3D Gaussians. # TODO: Verify shape
            quats                   :   A torch.Tensor of shape (N, 4) representing the rotation
                                        components of 3D Gaussians in quaternion form.
            scales                  :   A torch.Tensor of shape (N, 1) (if isotropic) or
                                        (N, 3) (if anisotropic) representing the scaling
                                        components of 3D Gaussians.
            colours                 :   A torch.Tensor of shape (N, 3) with the colour contribution
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
        means_2D = Gaussians.compute_means_2D(means_3D,camera)  # (N, 2)
        if not no_grad:
            means_2D.retain_grad()

        cov_2D = self.gaussians.compute_cov_2D(means_3D,quats,scales,camera,img_size)  # (N, 2, 2)

        # 计算每个片元投影后在图片上的半径（2*2协方差矩阵的特征值）
        mid = 0.5*(cov_2D[:,0,0] + cov_2D[:,1,1])
        det = cov_2D[:,0,0] * cov_2D[:,1,1] - cov_2D[:,1,0]*cov_2D[:,0,1]
        temp=torch.Tensor([0.1]).to(det.device)
        lambda1 = mid + torch.sqrt(torch.max(temp, mid * mid - det))
        lambda2 = mid - torch.sqrt(torch.max(temp, mid * mid - det))
        radii = torch.ceil(3.0* torch.sqrt(torch.max(lambda1, lambda2))) # (N,1)

        # Step 2: Compute alpha maps for each gaussian

        alphas = self.compute_alphas(opacities,means_2D,cov_2D,img_size)  # (N, H, W)

        # Step 3: Compute transmittance maps for each gaussian

        transmittance = self.compute_transmittance(alphas,start_transmittance)  # (N, H, W)

        # Some unsqueezing to set up broadcasting for vectorized implementation.
        # You can selectively comment these out if you want to compute things
        # in a diferent way.
        z_vals = z_vals[:, None, None, None]  # (N, 1, 1, 1)
        alphas = alphas[..., None]  # (N, H, W, 1)
        colours = colours[:, None, None, :]  # (N, 1, 1, 3)
        transmittance = transmittance[..., None]  # (N, H, W, 1)

        # Step 4: Create image, depth and mask by computing the colours for each pixel.

        image = torch.sum(colours*alphas*transmittance,dim=0)  # (H, W, 3)

        depth = torch.sum(z_vals*alphas*transmittance,dim=0)  # (H, W, 1)

        mask = torch.sum(alphas*transmittance,dim=0)  # (H, W, 1)

        final_transmittance = transmittance[-1, ..., 0].unsqueeze(0)  # (1, H, W)
        return image, depth, mask, final_transmittance,means_2D,radii

    def render(
        self, camera: PerspectiveCameras,
        per_splat: int = -1, img_size: Tuple = (256, 256),
        bg_colour: Tuple = (0.0, 0.0, 0.0),
        no_grad=False
    ):
        """
        Given a scene represented by N 3D Gaussians, this function renders the RGB
        colour image, the depth map and the silhouette map that can be observed
        from a given pytorch 3D camera.

        Args:
            camera      :   A pytorch3d PerspectiveCameras object.
            per_splat   :   Number of gaussians to splat in one function call. If set to -1,
                            then all gaussians in the scene are splat in a single function call.
                            If set to any other positive interger, then it determines the number of
                            gaussians to splat per function call (the last function call might splat
                            lesser number of gaussians). In general, the algorithm can run faster
                            if more gaussians are splat per function call, but at the cost of higher GPU
                            memory consumption.
            img_size    :   The (width, height) of the image to be rendered.
            bg_color    :   A tuple indicating the RGB colour that the background should have.

        Returns:
            image       :   A torch.Tensor of shape (H, W, 3) with the rendered RGB colour image.
            depth       :   A torch.Tensor of shape (H, W, 1) with the rendered depth map.
            mask        :   A torch.Tensor of shape (H, W, 1) with the rendered silhouette map.
        """
        bg_colour_ = torch.tensor(bg_colour)[None, None, :]  # (1, 1, 3)
        bg_colour_ = bg_colour_.to(self.device)

        # Globally sort gaussians according to their depth value
        z_vals = self.compute_depth_values(camera)
        idxs = self.get_idxs_to_filter_and_sort(z_vals)

        pre_act_quats = self.gaussians.pre_act_quats[idxs]
        pre_act_scales = self.gaussians.pre_act_scales[idxs]
        pre_act_opacities = self.gaussians.pre_act_opacities[idxs]
        z_vals = z_vals[idxs]
        means_3D = self.gaussians.means[idxs]

        # For questions 1.1, 1.2 and 1.3.2, use the below line of code for colours.
        if not self.gaussians.sphere:
            colours = self.gaussians.get_colour[idxs] # 不用球谐分量

        # [Q 1.3.1] For question 1.3.1, uncomment the below three lines to calculate the
        # colours instead of using self.gaussians.colours[idxs]. You may also comment
        # out the above line of code since it will be overwritten anyway.
        # 使用球谐分量
        else:
            spherical_harmonics = self.gaussians.spherical_harmonics[idxs]
            gaussian_dirs = self.calculate_gaussian_directions(means_3D, camera)
            colours = colours_from_spherical_harmonics(spherical_harmonics, gaussian_dirs)

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
            image, depth, mask, _,means_2D,radii = self.splat(
                camera, means_3D, z_vals, quats, scales,
                colours, opacities, img_size,no_grad=no_grad
            )

        # In this case we splat per_splat number of gaussians per iteration. This makes
        # the implementation more memory efficient but at the same time makes it slower.
        else:

            W, H = img_size
            D = means_3D.device
            start_transmittance = torch.ones((1, H, W), dtype=torch.float32).to(D)
            image = torch.zeros((H, W, 3), dtype=torch.float32).to(D)
            depth = torch.zeros((H, W, 1), dtype=torch.float32).to(D)
            mask = torch.zeros((H, W, 1), dtype=torch.float32).to(D)
            radii=torch.zeros((len(means_3D), 1), dtype=torch.float32).to(D)
            means_2D_list=[]

            # 每次计算一部分片元的颜色，最后叠加起来得到总的结果
            for b_idx in range(num_mini_batches):

                quats_ = quats[b_idx * per_splat: (b_idx+1) * per_splat]
                scales_ = scales[b_idx * per_splat: (b_idx+1) * per_splat]
                z_vals_ = z_vals[b_idx * per_splat: (b_idx+1) * per_splat]
                colours_ = colours[b_idx * per_splat: (b_idx+1) * per_splat]
                means_3D_ = means_3D[b_idx * per_splat: (b_idx+1) * per_splat]
                opacities_ = opacities[b_idx * per_splat: (b_idx+1) * per_splat]

                # Get image, depth and mask via splatting
                image_, depth_, mask_, start_transmittance,means_2D_,radii_ = self.splat(
                    camera, means_3D_, z_vals_, quats_, scales_, colours_,
                    opacities_, img_size, start_transmittance,no_grad=no_grad
                ) # 这里means_2D没有累加，此模式运行有问题，必须一次全导入

                image = image + image_
                depth = depth + depth_
                mask = mask + mask_
                radii[b_idx * per_splat: (b_idx+1) * per_splat]=radii_[:,None]
                means_2D_list.append(means_2D_)

            means_2D=torch.cat(means_2D_list,dim=0)

        image = mask * image + (1.0 - mask) * bg_colour_

        return image, depth, mask,means_2D,radii
