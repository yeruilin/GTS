function [FV]=VoxelSlicesPlotter(VoxelMat,thresh,step)
%detect the external voxels and faces
vol_handle=0;

FV.vertices=zeros(4*size(VoxelMat,1)*size(VoxelMat,2)*size(VoxelMat,3),3); %顶点坐标，每个坐标是三维
FV.faces=zeros(size(VoxelMat,1)*size(VoxelMat,2)*size(VoxelMat,3),4); %每个面四个顶点，第二维是4
FV.FaceVertexCData=zeros(size(VoxelMat,1)*size(VoxelMat,2)*size(VoxelMat,3),3); %颜色列表
FaceIndex=1;
VertexIndex=0;
counter=1; % 顶点计数
ExternalIndexes=zeros(size(VoxelMat,1)*size(VoxelMat,2)*size(VoxelMat,3),3); %不重复地保存顶点

for i=1:step:size(VoxelMat,1)
    for j=1:size(VoxelMat,2)
        for k=1:size(VoxelMat,3)
            if VoxelMat(i,j,k)>thresh
                % 计算顶点坐标
                ExternalIndexes(counter,1:3)=[i/step j k];
                FV.vertices(VertexIndex+1:VertexIndex+4,:)=[ExternalIndexes(counter,:)+[0 0 0]; ...
                    ExternalIndexes(counter,:)+[0 1 0];ExternalIndexes(counter,:)+[0 1 1];ExternalIndexes(counter,:)+[0 0 1]];
                
                % 新增一个面
                FV.faces(FaceIndex,:)=[FaceIndex*4-3 FaceIndex*4-2 FaceIndex*4-1 FaceIndex*4];
                FV.FaceVertexCData(FaceIndex,:)=[VoxelMat(i,j,k) VoxelMat(i,j,k) VoxelMat(i,j,k)];
                FaceIndex=FaceIndex+1;
                
                counter=counter+1;
                VertexIndex=VertexIndex+4;
            end
        end
    end 
end

counter=counter-1;
FV.vertices=FV.vertices(any(FV.vertices,2),:); %获取哪些位置有数值而不是0保留下来
FV.faces=FV.faces(any(FV.faces,2),:);
FV.FaceVertexCData=FV.FaceVertexCData(any(FV.FaceVertexCData,2),:);

cla;
if size(FV.vertices,1)==0
    cla;
else
vol_handle=patch('Vertices',FV.vertices,'Faces',FV.faces,'FaceVertexCData',FV.FaceVertexCData,'FaceColor','flat','EdgeColor','none');
%vol_handle=patch('Vertices',FV.vertices,'Faces',FV.faces,'FaceColor','r','EdgeColor','none');
end
end
