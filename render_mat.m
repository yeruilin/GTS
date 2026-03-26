clear;

load("temp/result.mat");
rho_=rho(5:end-5,5:end-5,5:end-5); % 去除边缘的噪声

rho_ = imgaussfilt3(rho_,2); % 这个参数可以调整
R=power(rho_,1);R=squeeze(max(R,[],3));

figure;imshow(R,[]);% colormap('hot');