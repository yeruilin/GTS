一、环境安装
conda create -n GTS python=3.12 # cu118安装python=3.10
conda activate GTS
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 # cu118
# pip install https://download.pytorch.org/whl/cu118/torch-2.6.0%2Bcu118-cp311-cp311-linux_x86_64.whl
pip install imageio matplotlib numpy PyMCubes tqdm scipy plotly plyfile opencv-python
pip install fvcore iopath
conda install -c bottler nvidiacub (linux不需要，windows安装方法不同)
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.6.0cu124 # pytorch3d==0.7.8+pt2.6.0cu118

二、代码说明：

train_confocal.py: 训练共焦场景，代码并没有用体渲染，但对单个墙面的非视域足够。 python train_confocal.py即可开始训练，5min内就可以收敛。

train_multi_view.py: 用体渲染方法做多视角重建。python train_multi_view.py即可开始训练，一般1-2min就可以收敛。

render_ply.py: 对ply格式表示的三维场景进行绘制。python rendeply.py --data_path temp/result.ply绘制结果。

render_mat.m: 对mat格式的三维场景进行绘制。将输出目录下的result.mat读取并执行就行。

gaussian.py: 高斯椭球的定义、增删策略，非体渲染的前向模型

scene.py: 非视域场景体渲染的实现

dataset.py: 读取数据，一般一个mat文件包括data(非视域数据), bin_resolution, width(完整墙宽)和t0(表示histogram第一个位置前飞行的时间)。不规则扫描点则是用grid描述采样点位置。

data_utils.py: 辅助函数

data/: 数据目录

results/: 几个结果示意图

三、提高效果的几个关键：

1. 增删椭球的策略：我们使用四叉树的方式增加椭球，而不是用Inria的版本，根据梯度来新增椭球。

2. 激活函数：透明度的激活函数从sigmoid改为平方，这对体渲染下的结果提升有很大帮助。不用的情况依然是sigmoid。

3. 完善的前向模型：模型是高斯分布和瑞利分布叠加，去掉其中一个效果会下降。

4. 分tile进行体渲染：对结果没有影响，但是可以大大降低显存占用。