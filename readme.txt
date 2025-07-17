conda create -n learning3d python=3.12 # cu118安装python=3.10
conda activate learning3d
pip install torch --index-url https://download.pytorch.org/whl/cu124 # cu118
# pip install https://download.pytorch.org/whl/cu118/torch-2.6.0%2Bcu118-cp311-cp311-linux_x86_64.whl
pip install imageio matplotlib numpy PyMCubes tqdm scipy plotly plyfile opencv-python
pip install fvcore iopath
conda install -c bottler nvidiacub (linux不需要，windows安装方法不同)
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.6.0cu124 # pytorch3d==0.7.8+pt2.6.0cu118