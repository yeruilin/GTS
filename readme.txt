conda create -n learning3d python=3.11
conda activate learning3d
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install fvcore iopath
conda install -c bottler nvidiacub (windows安装方法不同)
MAX_JOBS=8 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install imageio matplotlib numpy PyMCubes tqdm scipy plotly plyfile opencv-python