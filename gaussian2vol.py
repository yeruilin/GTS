import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
import argparse
from data_utils import load_gaussians_from_ply

def gaussians_to_voxel(gaussians, imgsize, radius, center):
    """
    将三维高斯集合转换为体素网格表示
    
    参数：
    gaussians - 高斯参数列表，每个元素为元组(mean, cov, intensity)
    
    返回：
    voxel_grid - 三维密度值数组
    """
    # 初始化体素网格
    bbox_min=np.array([center[0]-radius[0],center[1]-radius[1],center[2]-radius[2]])

    voxel_size=np.array([radius[0]*2/imgsize[0],radius[1]*2/imgsize[1],0]) # 每个网格大小
    voxel_size[2]=min(voxel_size[1],voxel_size[1])

    grid_size=[imgsize[0],imgsize[1],int(2*radius[2]/voxel_size[2])] # 三维网格大小

    print("grid_size:",grid_size)

    voxel = np.zeros(grid_size)

    # # 生成体素网格坐标（中心点坐标）
    # x = np.arange(grid_size[0]) * voxel_size[0] + bbox_min[0] + voxel_size[0]/2
    # y = np.arange(grid_size[1]) * voxel_size[1] + bbox_min[1] + voxel_size[1]/2
    # z = np.arange(grid_size[2]) * voxel_size[2] + bbox_min[2] + voxel_size[2]/2
    
    # # 预计算所有体素中心点坐标
    # xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    # positions = np.stack((xx, yy, zz), axis=-1)  # 形状为 (nx, ny, nz, 3)

    means, sigmas, intensity=gaussians

    # 遍历所有高斯分布
    for i in range(means.shape[0]):
        # 计算3σ边界
        min_corner = means[i,:] - 3*sigmas[i]
        max_corner = means[i,:] + 3*sigmas[i]
        
        # 转换为体素索引范围
        idx_min = ((min_corner - bbox_min) // voxel_size).astype(int)
        idx_max = ((max_corner - bbox_min) // voxel_size).astype(int) + 1
        
        # 限制在网格范围内
        idx_min = np.clip(idx_min, 0, None)
        idx_max = np.clip(idx_max, None, grid_size)
        
        # 只计算影响区域内的体素
        voxel[idx_min[0]:idx_max[0], idx_min[1]:idx_max[1],idx_min[2]:idx_max[2]]+=1 # intensity[i]

    return voxel

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path", default="./output", type=str,
        help="Path to the directory where output should be saved to."
    )
    parser.add_argument(
        "--data_path", default="./temp/result.ply", type=str,
        help="Path to the pre-trained gaussian data to be rendered."
    )
    parser.add_argument(
        "--img_dim", default=128, type=int,
        help=(
            "Spatial dimension of the rendered image. "
            "The rendered image will have img_dim as its height and width."
        )
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()

    ## 导入数据
    data_dict=load_gaussians_from_ply(args.data_path)
    means=data_dict["xyz"] # (N,3)
    sigma=np.exp(data_dict["scale"]) # (N,1)
    # sigma=np.ones((means.shape[0],))*0.01
    intensity=data_dict["dc_colours"][:,0]**2 # (N,1)


    # radius=[0.4,0.4,0.2] ## mannequin数据的参数
    # center=(0.0,0.0,0.55)
    # radius=[0.5,0.5,0.4] ## bunny的参数
    # center=(0.0037,0.1018,0.8335)
    radius=[0.3,0.3,0.3] ## teapot数据的参数 
    center=(0.0821,0.2270,1.1992)
    # radius=[1.0,1.0,0.3] ## fk-bike数据参数
    # center=(0.0,0.15,1.35)
    # radius=[1.0,1.0,0.5] ## fk-dragon数据参数
    # center=(-0.1,0.1,1.35)
    # radius=[1.2,1.2,0.6] ## fk-teaser数据参数
    # center=(0.0,0.0,1.35)

    # 高斯转体素
    voxel=gaussians_to_voxel([means,sigma,intensity],[args.img_dim,args.img_dim],radius,center)

    # # 平滑处理
    # sigma = 1.0  # 定义高斯核的标准差（sigma）
    # filtered_vol = gaussian_filter(voxel, sigma=sigma, mode='constant', truncate=2.5)
    
    scipy.io.savemat("temp/voxel.mat",{"voxel":voxel})

    print("Finish!")



