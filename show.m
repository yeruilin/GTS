% load("temp/voxel.mat");


%% 绘制矩形框
x_inner1=0;
y_inner1=0;
z_inner1=0;
x_inner2=size(voxel,1);
y_inner2=size(voxel,2);
z_inner2=size(voxel,3);
gridsize=size(voxel,1);
%顶点矩阵;
vertex_matrix=[x_inner2 y_inner2 z_inner2;x_inner1 y_inner2 z_inner2;x_inner1 y_inner1 z_inner2;x_inner2 y_inner1 z_inner2;
               x_inner2 y_inner2 z_inner1;x_inner1 y_inner2 z_inner1;x_inner1 y_inner1 z_inner1;x_inner2 y_inner1 z_inner1];
 
%连接矩阵：连接关系矩阵每一行中的数值分别表示顶点矩阵的行标;
face_matrix=[1 2 6 5;2 3 7 6;3 4 8 7;
             4 1 5 8;1 2 3 4;5 6 7 8];

%% 开始绘制
step=1;
for ii=1:1
    [vol_handle]=VoxelSlicesPlotter(voxel,0.0,step); %绘制切片
    %绘制方框
    patch('Vertices',vertex_matrix,'Faces',face_matrix,'FaceVertexCData',hsv(8),'FaceColor','none','EdgeColor','r');
    view(3);
    daspect([1,1,1]);
    set(gcf,'color','black'); %窗口背景黑色
    colordef black; %patch 2D/3D图背景黑色
    set(gca,'xlim',[0 gridsize], 'ylim',[0 gridsize], 'zlim',[0 gridsize],'xcolor','none','ycolor','none','zcolor','none');
    
    hold on;
    plot3([ii,ii],[0,0],[0,gridsize],'yellow');
    
    % title("measurements",'color','white');
    
    pause(0.1);
    set(gcf, 'InvertHardCopy', 'off'); %保存背景色
    % saveas(gcf,['measurements\',num2str(ii),'.tif']);
end