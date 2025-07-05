import math
import torch
import numpy as np

from typing import Tuple, Optional
from gaussian import Gaussians
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.transforms import quaternion_to_matrix

class Scene:
    def __init__(self, gaussians: Gaussians):
        self.gaussians = gaussians
        self.device = self.gaussians.device
    
    def compute_cov_3D(self, quats: torch.Tensor, scales: torch.Tensor, is_isotropic=True):
        """
        Computes the covariance matrices of 3D Gaussians using equation (6) of the 3D
        Gaussian Splatting paper.

        Link: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf

        Args:
            quats   :   A torch.Tensor of shape (N, 4) representing the rotation
                        components of 3D Gaussians in quaternion form.
            scales  :   If is_isotropic is True, scales is will be a torch.Tensor of shape (N, 1)
                        If is_isotropic is False, scales is will be a torch.Tensor of shape (N, 3).
                        Represents the scaling components of the 3D Gaussians.

        Returns:
            cov_3D  :   A torch.Tensor of shape (N, 3, 3)
        """

        if is_isotropic:
            scales=scales*scales
            scales=scales.repeat((1,3))
            cov_3D=torch.diag_embed(scales) # (N, 3, 3)
        else:
            R=quaternion_to_matrix(quats) # (N,3,3)
            scales=scales*scales
            cov_3D=torch.diag_embed(scales) # (N, 3, 3)
            cov_3D = torch.bmm(R,torch.bmm(cov_3D,R.transpose(1,2)))  # (N, 3, 3)

        return cov_3D

    def _compute_jacobian(self, means_3D: torch.Tensor, camera, img_size: Tuple):

        if type(camera)==PerspectiveCameras:
            if camera.in_ndc():
                raise RuntimeError
            
            fx, fy = camera.focal_length.flatten()
            W, H = img_size

            half_tan_fov_x = 0.5 * W / fx
            half_tan_fov_y = 0.5 * H / fy
        
        else: # FoVCamera
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
        J = J.to(self.device)

        J[:, 0, 0] = fx / tz
        J[:, 1, 1] = fy / tz
        J[:, 0, 2] = -(fx * tx) / tz2
        J[:, 1, 2] = -(fy * ty) / tz2

        return J  # (N, 2, 3)
    
    def compute_cov_2D(
        self, means_3D: torch.Tensor, quats: torch.Tensor, scales: torch.Tensor,
        camera: PerspectiveCameras, img_size: Tuple
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
            camera      :   A pytorch3d PerspectiveCameras object
            img_size    :   A tuple representing the (width, height) of the image

        Returns:
            cov_3D  :   A torch.Tensor of shape (N, 3, 3)
        """
        N=quats.shape[0]

        J = self._compute_jacobian(means_3D,camera,img_size)  # (N, 2, 3)

        W = camera.R.repeat((N,1,1))  # (N, 3, 3)

        cov_3D = self.compute_cov_3D(quats,scales)  # (N, 3, 3)

        J=torch.bmm(J,W)
        cov_2D = torch.bmm(J,torch.bmm(cov_3D,J.transpose(1,2)))  # (N, 2, 2)

        # Post processing to make sure that each 2D Gaussian covers atleast approximately 1 pixel
        cov_2D[:, 0, 0] += 0.3
        cov_2D[:, 1, 1] += 0.3

        return cov_2D

    @staticmethod
    def compute_means_2D(means_3D: torch.Tensor, camera: PerspectiveCameras):
        """
        Computes the means of the projected 2D Gaussians given the means of the 3D Gaussians.

        Args:
            means_3D    :   A torch.Tensor of shape (N, 3) representing the means of
                            3D Gaussians.
            camera      :   A pytorch3d PerspectiveCameras object.

        Returns:
            means_2D    :   A torch.Tensor of shape (N, 2) representing the means of
                            2D Gaussians.
        """

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
        vec=points_2D-means_2D # (N,H*W,2)
        vec2=torch.bmm(cov_2D_inverse,vec.transpose(1,2)).transpose(1,2) # (N,H*W,2)
        power =vec[:,:,0]*vec2[:,:,0]+vec[:,:,1]*vec2[:,:,1]  # (N, H*W)

        return -0.5*power

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

        cov_2D_inverse = self.invert_cov_2D(cov_2D)  # (N, 2, 2) TODO: Verify shape

        power = self.evaluate_gaussian_2D(points_2D,means_2D,cov_2D_inverse)  # (N, H*W)

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
        means_2D = self.compute_means_2D(means_3D,camera)  # (N, 2)
        if not no_grad:
            means_2D.retain_grad()

        cov_2D = self.compute_cov_2D(means_3D,quats,scales,camera,img_size)  # (N, 2, 2)

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
        
        return image, depth, mask, final_transmittance

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
        colours = self.gaussians.get_colour[idxs]
        quats = torch.zeros([colours.shape[0],4],dtype=colours.dtype,device=colours.device)
        quats[:,3]=1.0
        scales = self.gaussians.get_scaling[idxs]
        opacities = self.gaussians.get_opacity[idxs].flatten()
        z_vals = z_vals[idxs]
        means_3D = self.gaussians.means[idxs]

        if per_splat == -1:
            num_mini_batches = 1
        elif per_splat > 0:
            num_mini_batches = math.ceil(len(means_3D) / per_splat)
        else:
            raise ValueError("Invalid setting of per_splat")

        # In this case we can directly splat all gaussians onto the image
        if num_mini_batches == 1:

            # Get image, depth and mask via splatting
            image, depth, mask,_ = self.splat(
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

            # 每次计算一部分片元的颜色，最后叠加起来得到总的结果
            for b_idx in range(num_mini_batches):

                quats_ = quats[b_idx * per_splat: (b_idx+1) * per_splat]
                scales_ = scales[b_idx * per_splat: (b_idx+1) * per_splat]
                z_vals_ = z_vals[b_idx * per_splat: (b_idx+1) * per_splat]
                colours_ = colours[b_idx * per_splat: (b_idx+1) * per_splat]
                means_3D_ = means_3D[b_idx * per_splat: (b_idx+1) * per_splat]
                opacities_ = opacities[b_idx * per_splat: (b_idx+1) * per_splat]

                # Get image, depth and mask via splatting
                image_, depth_, mask_, start_transmittance = self.splat(
                    camera, means_3D_, z_vals_, quats_, scales_, colours_,
                    opacities_, img_size, start_transmittance,no_grad=no_grad
                ) # 这里means_2D没有累加，此模式运行有问题，必须一次全导入

                image = image + image_
                depth = depth + depth_
                mask = mask + mask_

        image = mask * image + (1.0 - mask) * bg_colour_

        return image, depth, mask
    
    def render_conf_hist(self,camera,bin_resolution,num_bins,t0=0,decay=4,gaussians_per_splat=-1,img_size=64):
        # Globally sort gaussians according to their depth value
        z_vals_origin = self.compute_depth_values(camera) # (N,)
        idxs = self.get_idxs_to_filter_and_sort(z_vals_origin)

        scales = self.gaussians.get_scaling[idxs] # [N,1]
        opacities = self.gaussians.get_opacity[idxs].view(-1) # [N,]
        quats= torch.zeros([scales.shape[0],4],dtype=scales.dtype,device=scales.device)
        z_vals = z_vals_origin[idxs] # [N,1]
        means_3D = self.gaussians.means[idxs] # [N,3]

        colours = self.gaussians.get_colour[idxs] # [N,1]

        ## Volume rendering histogram
        N=means_3D.shape[0]
        if z_vals.shape[0]!=N:
            raise RuntimeError
        # Step 1: Compute 2D gaussian parameters
        means_2D = self.compute_means_2D(means_3D,camera)  # (N, 2)
        cov_2D = self.compute_cov_2D(means_3D,quats,scales,camera,img_size)  # (N, 2, 2)

        # Step 2: Compute alpha maps for each gaussian
        alphas = self.compute_alphas(opacities,means_2D,cov_2D,img_size)  # (N, H, W)

        # Step 3: Compute transmittance maps for each gaussian
        transmittance = self.compute_transmittance(alphas)  # (N, H, W)

        # integrate on phi and theta
        intensity=colours[:,:,None]*alphas*transmittance  # (N,H, W)
        intensity = torch.sum(intensity.view(N, -1), dim=1)  # (N,)

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