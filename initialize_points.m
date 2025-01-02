% 将瞬态图转换为points点云
% 这里默认左下角是(-half_width,-half_width)的位置，即便扫描顺序不是从左下角开始，结果只是发生了xy的旋转。
% 因此不影响整个Gaussian Transient Splatting的优化。无非是优化的结果也是需要旋转90°或者180°复位。

tic;
sig=double(data);
% width=2.5;
half_width=width/2;
% Confocal Non-Light-of-Sight (C-NLOS) reconstruction procedure for paper
% titled "Confocal Non-Line-of-Sight Imaging Based on the Light Cone Transform"
% by Matthew O'Toole, David B. Lindell, and Gordon Wetzstein.
%
% Constants
c              = 3e8; % Speed of light (meters per second)
% bin_resolution = 0.02/c; % Native bin resolution for SPAD is 4 ps
  

% Adjustable parameters
isbackprop = 0;         % Toggle backprojection
isdiffuse  = 0;         % Toggle diffuse reflection
snr        = 0.1;      % SNR value

N = size(sig,2);        % Spatial resolution of data，空间采样点数为N*N
M = size(sig,3);        % Temporal resolution of data,也就是时间维度采样数目
range = M.*c.*bin_resolution; % Maximum range for histogram

% Define NLOS blur kernel 
psf = definePsf(N,M,half_width/range);%1024*128*128

% Compute inverse filter of NLOS blur kernel
fpsf = fftn(psf);
% temp1=squeeze(fpsf(2,1,:));
% if (~isbackprop)
   invpsf = conj(fpsf) ./ (abs(fpsf).^2 + 1./snr);
% else
%     invpsf = conj(fpsf);
% end

% Define transform operators
[mtx,mtxi] = resamplingOperator(M);
% 
% % Permute data dimensions
sig = permute(sig,[3 2 1]);

vol = LCT(sig,M,N,fpsf,mtx,mtxi,snr);
% vol=FK(sig,M,N,range,half_width); % 效果不好

threedshow(vol,range,half_width);

%%% 生成点云
thresh=1e-5;%5e-6
step=(half_width*2)/(N-1);
vol2 = permute(vol,[3 2 1]); % 重新变为XYZ顺序
% LCT用1e-5，FK用1.0，选出5000个点的阈值就行了
% 找到所有大于 0 的位置
[xindex, yindex, zindex] = ind2sub(size(vol2), find(vol2 > thresh));

x = -half_width+step*(xindex-1);
y = -half_width+step*(yindex-1);
z = zindex*(c*bin_resolution/2);%Z方向的距离为2m
points=[x,y,z];

function psf = definePsf(U,V,slope)
% Local function to computeD NLOS blur kernel
% 划分空间网格
x = linspace(-1,1,2.*U);
y = linspace(-1,1,2.*U);
z = linspace(0,2,2.*V);%Z方向的距离为2m
[grid_z,grid_y,grid_x] = ndgrid(z,y,x);

% Define PSF
psf = abs(((4.*slope).^2).*(grid_x.^2 + grid_y.^2) - grid_z);

test00=squeeze(psf(:,1,2));

% min(psf,[],1)表示返回1*2U*2U的张量，计算出时间上的最小值
% repeat(A,[2.*V 1 1])指A的一维扩充到2V个，然后其他维度保持不变
% ==返回一个逻辑数组，然后转成double类型
%找到最接近于0的时间作为δ函数取到的点，对于每个psf(:,i,j)，都只有一个时刻为1
psf = double(psf == repmat(min(psf,[],1),[2.*V 1 1]));
psf = psf./sum(psf(:,U,U));%sum(psf(:,U,U))=1，所以并不太需要在这里进行归一化
psf = psf./norm(psf(:));%norm(psf(:))=U*U，也就是点的数目
psf = circshift(psf,[0 U U]);%每个维度循环地移动，也就是x=(x+0)%xlen;y=(y+U)%ylen;z=(z+U)%zlen
%最后这一步应该是为了适配matlab的fftn，所以做了一个平移
end

% 第一个是Rt,第二个是Rz^{-1}，理论上大小都是M^2*M，M^2行里面都只有M行有数值，所以是稀疏矩阵
% Rt将原本的τ(x,y,t)变成v^(3/2)τ(x,y,v)
% Rz^{-1}将1/2\sqrt(u)\rho(x,y,sqrt(u))映射到\rho(x,y,z)
% 但是这个代码做了大幅优化，既然只有M行是有效，不如直接压缩成M*M，于是返回的两个矩阵都是M*M
% 不过这要求M必须是2的整数倍
% 但是Rt和Rz^{-1}应该很不一样，但是因为原始信号在第一步乘上了v^2，导致v^(3/2)τ(x,y,v)变成了1/\sqrt(v)τ(x,y,v)
% 这样使得两个矩阵变得有着惊人的相似了
% 成像本来是(x,y,z)->(x',y',t)的映射，这里的重采样算子变成了(x,y,u)->(x',y',v)的映射
function [mtx,mtxi] = resamplingOperator(M)
% Local function that defines resampling operators

%S=sparse(i,j,s,m,n,num) 由向量i,j,s生成一个m*n的含有num个非零元素的稀疏矩阵S
mtx = sparse([],[],[],M.^2,M,M.^2);%M^2*M的全零稀疏矩阵

x = 1:M.^2;
%sub2ind是给出mtx矩阵的大小(a,b)，给出行列索引(i,j)，返回对应的索引值，本质上就是i*b+j
mtx(sub2ind(size(mtx),x,ceil(sqrt(x)))) = 1;%访问mtx的(x,ceil(sqrt(x)))的元素
%A = spdiags(B,d,m,n)产生一个m*n的稀疏矩阵根据B的每一列，并将其他位置填上d
%这里就是得到主对角线元素为1./sqrt(x)'的M^2*M^2的稀疏矩阵
mtx  = spdiags(1./sqrt(x)',0,M.^2,M.^2)*mtx;%mtx的每一行除以sqrt(x)
mtxi = mtx';

% 对M^2*M的矩阵压缩到M*M
K = log(M)./log(2);%计算log2(M)
for k = 1:round(K)
    mtx  = 0.5.*(mtx(1:2:end,:)  + mtx(2:2:end,:));%相邻两行取平均合并
    mtxi = 0.5.*(mtxi(:,1:2:end) + mtxi(:,2:2:end));
end
end