### 测试分布式训练

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os

class GaussianModel(nn.Module):
    def __init__(self, num_gaussians):
        super().__init__()
        self.xyz = nn.Parameter(torch.randn(num_gaussians, 3))
        self.rho = nn.Parameter(torch.randn(num_gaussians, 1))
        self.o = nn.Parameter(torch.randn(num_gaussians, 1))

    def parameters(self):
        return [self.xyz, self.rho, self.o]
    
    def get_all_parameters(self):
        """返回当前GPU上的所有Gaussian参数"""
        return {
            'xyz': self.xyz.detach(),
            'rho': self.rho.detach(),
            'o': self.o.detach()
        }
    
    def forward(self,x): # forward函数一定要有输入
        hist=torch.sum(self.xyz, dim=1, keepdim=True) *self.rho * self.o+x
        return hist

def setup(rank, world_size):
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12345',
        rank=rank,
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()


def gather_all_parameters(rank, world_size, local_params):
    """
    收集所有GPU上的参数到主GPU
    返回: 主GPU上包含所有参数的字典
    """
    # 为每个参数创建存储列表
    xyz_list = [torch.zeros_like(local_params['xyz']) for _ in range(world_size)]
    rho_list = [torch.zeros_like(local_params['rho']) for _ in range(world_size)]
    o_list = [torch.zeros_like(local_params['o']) for _ in range(world_size)]
    
    # 收集所有GPU的参数
    dist.all_gather(xyz_list, local_params['xyz'])
    dist.all_gather(rho_list, local_params['rho'])
    dist.all_gather(o_list, local_params['o'])
    
    if rank == 0:
        # 在主GPU上拼接所有参数
        all_params = {
            'xyz': torch.cat(xyz_list, dim=0).cpu(),
            'rho': torch.cat(rho_list, dim=0).cpu(),
            'o': torch.cat(o_list, dim=0).cpu()
        }
        return all_params
    return None

def save_parameters(all_params, save_dir="saved_gaussians"):
    """保存所有参数到本地"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存为PyTorch tensor文件
    torch.save(all_params['xyz'], os.path.join(save_dir, "xyz.pt"))
    torch.save(all_params['rho'], os.path.join(save_dir, "rho.pt"))
    torch.save(all_params['o'], os.path.join(save_dir, "opacity.pt"))
    
    # 也可以保存为numpy格式
    numpy_params = {k: v.numpy() for k, v in all_params.items()}
    torch.save(numpy_params, os.path.join(save_dir, "gaussians_all.npz"))
    
    print(f"All parameters saved to {save_dir}")

def train(rank, world_size, total_gaussians, hist_gt,resume=False):
    setup(rank, world_size)
    
    # 每个GPU处理的数据量
    num_gaussians_per_gpu = total_gaussians // world_size
    
    # 创建模型并移动到当前GPU
    model = GaussianModel(num_gaussians_per_gpu).to(rank)

    # 导入分布式下的权重
    if resume:
        model.load_state_dict(torch.load("model.pt", map_location=f"cuda:{rank}"))

    ddp_model = DDP(model, device_ids=[rank])
    
    # 优化器
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    
    # 将真实结果复制到当前GPU
    hist_gt = hist_gt.to(rank)
    
    for epoch in range(100):  # 假设训练100个epoch
        optimizer.zero_grad()
        
        # 前向传播，计算当前GPU的hist_pred
        x=torch.ones(num_gaussians_per_gpu,1).to(rank)
        hist_pred_local = ddp_model(x)
        loss=hist_pred_local - hist_gt

        # DDP的all reduce方法将每个节点的hist传递到其他节点并完成求和。
        # 但是只需要在主节点计算一次，DDP就会自动回传到每个节点并更新。
        # 当然其他节点也需要在形式上完成一个回传，因此可以用dummy loss回传即可
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)  # now it's total_hist_pred
        ## 但要注意，DDP本身是不支持all_reduce的自动微分，因此这里会warning！而且用在高斯泼墨中会有问题！！！

        # print(hist_pred_local.shape) # (2500,1)
        if rank == 0:
            loss = torch.mean(loss**2)/world_size
            print(f"Epoch {epoch} | Loss: {loss.item()}")
            loss.backward()
        else:
            # Dummy loss for backward compatibility
            dummy_loss = (hist_pred_local.sum() * 0)
            dummy_loss.backward()

        optimizer.step()
        
        if rank == 0 and epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    # 将所有参数汇聚到主节点一起保存
    local_params = model.get_all_parameters()
    all_params = gather_all_parameters(rank, world_size, local_params)
    if rank == 0:
        save_parameters(all_params)

    # # 按照分布权重的方法保存
    # if rank == 0: # 在主节点保存一次即可
    #     torch.save(ddp_model.module.state_dict(), "model.pt")

    cleanup()

if __name__ == "__main__":
    # 配置参数
    world_size = 4  # GPU数量
    total_gaussians = 10000  # 总Gaussian对象数
    
    # 生成模拟的真实hist_gt (示例数据，替换为你的实际数据)
    hist_gt = torch.randn(2500 * world_size)  # 假设hist_pred的长度是2500
    
    # 使用多进程启动训练
    mp.spawn(
        train,
        args=(world_size, total_gaussians, hist_gt),
        nprocs=world_size,
        join=True
    )