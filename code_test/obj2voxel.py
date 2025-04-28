import trimesh
import scipy
import numpy as np

def process_obj_to_voxel(obj_path, voxel_res=64, target_size=0.6):
    """
    处理OBJ文件并生成体素网格
    参数:
        obj_path: OBJ文件路径
        voxel_res: 体素分辨率（每个维度的体素数）
        target_size: 目标包围盒边长
    返回:
        voxel_grid: 三维二值体素数组
    """
    # 加载网格
    mesh = trimesh.load(obj_path)
    
    # Step 1: 平移物体中心到原点
    # --------------------------------------
    # 计算原始包围盒中心
    centroid = mesh.bounding_box.centroid
    # 平移顶点使中心到原点
    mesh.vertices -= centroid
    
    # Step 2: 缩放到目标尺寸
    # --------------------------------------
    # 计算当前包围盒尺寸
    current_size = mesh.bounding_box.extents.max()
    # 计算缩放比例
    scale_factor = target_size / current_size
    # 应用均匀缩放
    mesh.vertices *= scale_factor
    
    # Step 3: 体素化
    # --------------------------------------
    # 计算包围盒新尺寸
    bbox_size = mesh.bounding_box.extents
    # 计算体素边长（保持各向同性）
    pitch = bbox_size.max() / voxel_res
    
    # 生成体素网格
    voxel = mesh.voxelized(pitch=pitch)
    # 转换为稠密数组
    dense_grid = voxel.matrix.astype(np.float32)
    
    return dense_grid

# 使用示例
if __name__ == "__main__":
    # 输入参数
    obj_path = "data/bunny.obj"  # 替换为你的OBJ文件路径
    voxel_resolution = 64    # 体素分辨率
    output_size = 0.6        # 目标包围盒边长
    
    # 执行处理
    voxel_grid = process_obj_to_voxel(obj_path, voxel_resolution, output_size)
    
    # 输出结果
    print(f"体素网格形状: {voxel_grid.shape}")
    print(f"非零体素数量: {np.sum(voxel_grid)}")
    
    # 保存结果（可选）
    scipy.io.savemat("voxel_grid.mat",{"voxel":voxel_grid})
    # np.save("voxel_grid.npy", voxel_grid)