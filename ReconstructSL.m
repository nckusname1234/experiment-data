clc,close all,warning off 
clearvars -except stereoParams 
prediction_folder_name = {'FEMENet'; 'Ground_Truth'; 'Original'; 'HDR'; 'SL_Inpainting'; 'SP_CAN'; 'UNet'};

%%
for obj_num_i=1:9
    obj_num = num2str(obj_num_i,'%02d');
    for n = 1:numel(prediction_folder_name)
        folder_name = char(prediction_folder_name(n));
        obj_path = strcat('./', obj_num, '/', folder_name);
        load('./calibration.mat') %load calibration parameter
        Reconstructdir = strcat('./', obj_num, '/PointCloud/',folder_name);
        level.col=10;
        level.row=9;
        projector.height=800;
        projector.width=1280;
        threshold=0;
        total_image_num = 40;
        % total_image_num=2*(level.col+level.row); %mean +2
        %-----------------------------------------------------
        tic;
        % GrayCodePatterns
        [Graycodepattern, binarycodepattern]=Graycoding(projector.height,projector.width,0,10);
        WidthPatterns=Graycodepattern(:,:,1:level.col); 
        HeightPatterns=Graycodepattern(:,:,level.col+1:level.col+level.row);
        % LargeGapGrayPatterns
        % [HeightPatterns,WidthPatterns]=LargeGapGrayPatterns(projector.height,projector.width);
        s=dir(obj_path);

        for i=1:total_image_num
            im=imread([obj_path,'\',s(i+2).name]);
            if size(im,3)==3
                im1(:,:,i)=rgb2gray(im);
            else
                im1(:,:,i)=im;
            end
            if i==1
                color=im;
            end
        end

        % fprintf(mat2str(size(im1)));
        M=im1(:,:,1)> im1(:,:,2)+threshold;
        shdr=double(im1(:,:,3:2:total_image_num)>im1(:,:,4:2:total_image_num));
        dc1=zeros(size(shdr,1),size(shdr,2));dc2=dc1;
        dp1=zeros(projector.height,projector.width);dp2=dp1;

        % dp1=Decode(WidthPatterns,level.col);
        % dp2=Decode(HeightPatterns,level.row);
        % dc1=Decode(shdr(:,:,1:level.col),level.col);
        % dc1=dc1.*double(M);
        % dc2=Decode(shdr(:,:,level.col+1:level.col+level.row),level.row);
        % dc2=dc2.*double(M);

        dp1=GrayDecode(Graycodepattern(:,:,1:level.col),level.col);
        dp2=GrayDecode(Graycodepattern(:,:,level.col+1:level.col+level.row),level.row);
        dc1=GrayDecode(shdr(:,:,1:level.col),level.col);
        dc1=dc1.*double(M);
        dc2=GrayDecode(shdr(:,:,level.col+1:level.col+level.row), level.row);
        dc2=dc2.*double(M);

        % figure,
        % subplot(2,2,1),imshow(dc1/2^level.col),title('camera decode result(col)'),colormap(jet)
        % subplot(2,2,2),imshow(dc2/2^level.row),title('camera decode result(row)'),colormap(jet)
        % subplot(2,2,3),imshow(dp1/2^level.col),title('projector decode result(col)'),colormap(jet)
        % subplot(2,2,4),imshow(dp2/2^level.row),title('projector decode result(row)'),colormap(jet)
        % pause(0.01)

        D1=cat(3,dc1,dc2);
        D2=cat(3,dp1,dp2);
        [Lm,Rm]=MapCodeCenter(D1,D2);
        % Lm=undistort_p(Lm,stereoParams.CameraParameters1);
        % Rm=undistort_p(Rm,stereoParams.CameraParameters2);
        % parfor i=1:size(Lm,1)
        %     Lm(i,:)= undistortPoints(Lm(i,:),stereoParams.CameraParameters1);
        %     Rm(i,:)= undistortPoints(Rm(i,:),stereoParams.CameraParameters2);
        % end
        Rm(isnan(Lm(:,1)),:)=[];Lm(isnan(Lm(:,1)),:)=[];
        %projector denoise
        RemoveMask=zeros(size(dp1));
        Lm(round(Rm(:,2))>=size(dp1,1),:)=[];Lm(round(Rm(:,2))<=0,:)=[];
        Rm(round(Rm(:,2))>=size(dp1,1),:)=[];Rm(round(Rm(:,2))<=0,:)=[];
        Lm(round(Rm(:,1))>=size(dp1,2),:)=[];Lm(round(Rm(:,1))<=0,:)=[];
        Rm(round(Rm(:,1))>=size(dp1,2),:)=[];Rm(round(Rm(:,1))<=0,:)=[];
        ind=sub2ind([size(dp1,1),size(dp1,2)],round(Rm(:,2)),round(Rm(:,1)));
        % RemoveMask(ind)=1;
        % RemoveMask2=ConnectRangeDenoise(RemoveMask,100);
        % Rm(RemoveMask2(ind)==1,:)=[];Lm(RemoveMask2(ind)==1,:)=[];
        figure,plot(Lm(:,1),Lm(:,2),'r*')
        % figure,plot(Rm(:,1),Rm(:,2),'r*')

        Rm(round(Lm(:,2))>=size(dc1,1),:)=[];Rm(round(Lm(:,2))<=0,:)=[];
        Lm(round(Lm(:,2))>=size(dc1,1),:)=[];Lm(round(Lm(:,2))<=0,:)=[];
        Rm(round(Lm(:,1))>=size(dc1,2),:)=[];Rm(round(Lm(:,1))<=0,:)=[];
        Lm(round(Lm(:,1))>=size(dc1,2),:)=[];Lm(round(Lm(:,1))<=0,:)=[];

        ind=sub2ind([size(dc1,1),size(dc1,2)], round(Lm(:,2)),round(Lm(:,1)));
        [p3d,d]=line_line_intersection(stereoParams,Lm,Rm);
        p3d(:,d>100)=[];ind(d>100)=[];
        if size(color,3)==3
            color1=color(:,:,1);color2=color(:,:,2);color3=color(:,:,3);
            pc=pointCloud(single(p3d'),'Color',[color1(ind),color2(ind),color3(ind)]);
        else
            pc=pointCloud(single(p3d'),'Color',[color(ind),color(ind),color(ind)]);
        end
        % pc=projectivereconstruct(stereoParams,[Lm,Rm],color);


        pc.Normal=pcnormals(pc);
        pc = pcdenoise(pc);
        pcshow(pc),set(gca,'color','y')
        pcwrite(pc,[Reconstructdir,'.ply'])
        strcat(Reconstructdir, '.ply')

    end
end  

function img=ConnectRangeDenoise(imBw,connect_num)
imLabel = bwlabel(imBw,8);
stats = regionprops(imLabel,'Area');
area = cat(1,stats.Area);
index = find(area<connect_num);
[img,~] = ismember(imLabel,index);
end

function [Lm,Rm]=MapCodeCenter(D1,D2)
D11=D1(:,:,1)+10000*D1(:,:,2);
D21=D2(:,:,1)+10000*D2(:,:,2);
code=intersect(D11(:),D21(:));
code(code<=0)=[];
code=code(randsample(length(code),length(code)));
[~,Locb1] = ismember(D11,code);
[~,Locb2] = ismember(D21,code);
J=regionprops(Locb1,'PixelList');
move=[0,0];
parfor i=1:length(J)
    h=J(i).PixelList;
    if size(h,1)<=5
        move=[move;h];
    else
        [~,tf]=rmoutliers(h,'median');
        move=[move;h(tf,:)];
    end
end
move(1,:)=[];
ind=sub2ind(size(Locb1),move(:,2),move(:,1));
Locb1(ind)=0;
code=intersect(Locb1(:),Locb2(:));code(code<=0)=[];
[~,Locb1] = ismember(Locb1,code);
[~,Locb2] = ismember(Locb2,code);
L_Centroid = regionprops(Locb1, 'Centroid');
R_Centroid = regionprops(Locb2, 'Centroid');
Lm=vertcat(L_Centroid.Centroid);
Rm=vertcat(R_Centroid.Centroid);
end

function [p3d,d]=line_line_intersection(stereoParams,Lpoint,Rpoint)

t1=1;% if projector col resolution > 1024 =>t1=2
t2=1;% if projector row resolution > 1024 =>t2=2
L_points=[Lpoint,ones(size(Lpoint,1),1)];
L_points(:,1)=(L_points(:,1)-stereoParams.CameraParameters1.PrincipalPoint(1))/stereoParams.CameraParameters1.FocalLength(1);
L_points(:,2)=(L_points(:,2)-stereoParams.CameraParameters1.PrincipalPoint(2))/stereoParams.CameraParameters1.FocalLength(2);
R_points=[Rpoint,ones(size(Rpoint,1),1)];
R_points(:,1)=(R_points(:,1)-stereoParams.CameraParameters2.PrincipalPoint(1))/stereoParams.CameraParameters2.FocalLength(1)/t1;
R_points(:,2)=(R_points(:,2)-stereoParams.CameraParameters2.PrincipalPoint(2))/stereoParams.CameraParameters2.FocalLength(2)/t2;
R=stereoParams.RotationOfCamera2;
T=stereoParams.TranslationOfCamera2;
d=zeros(size(L_points,1),1);
for i=1:size(L_points,1)
    w1=L_points(i,:)';
    w2=R*[R_points(i,:)'-T'];
    v1=w1;
    v2=R*R_points(i,:)';
    [p,d(i)]=approximate_ray_intersection(v1,w1,v2,w2);
    p3d(:,i)=R'*(p-T'); %Translate camera(L) coordinate to camera(R)
end

end

function [p,d]=approximate_ray_intersection(v1,q1,v2,q2)

v1tv1=v1'*v1;
v2tv2=v2'*v2;
v1tv2=v1'*v2;
v2tv1=v2'*v1;
detV=v1tv1*v2tv2 - v1tv2*v2tv1;
Vinv=[v2tv2/detV,v1tv2/detV;v2tv1/detV,v1tv1/detV];
q2_q1 = q2 - q1;
Q1 = v1(1)*q2_q1(1) + v1(2)*q2_q1(2) + v1(3)*q2_q1(3);
Q2 = -(v2(1)*q2_q1(1) + v2(2)*q2_q1(2) + v2(3)*q2_q1(3));
lambda1 = (v2tv2 * Q1 + v1tv2 * Q2) /detV;
lambda2 = (v2tv1 * Q1 + v1tv1 * Q2) /detV;
p1 = lambda1*v1 + q1; 
p2 = lambda2*v2 + q2; 
p = 0.5*(p1+p2);
d=norm(p1-p2);
end  

function pc=projectivereconstruct(stereoParams,mpix,colorp)
    P1=cameraMatrix(stereoParams.CameraParameters1,stereoParams.CameraParameters1.RotationMatrices(:,:,1),stereoParams.CameraParameters1.TranslationVectors(1,:))';
    P2=cameraMatrix(stereoParams.CameraParameters2,stereoParams.CameraParameters2.RotationMatrices(:,:,1),stereoParams.CameraParameters2.TranslationVectors(1,:))';
    color=[0,0,0];
    reconstruction=[0;0;0];
    for i=1:size(mpix,1)
        A=[P1(3,1)*mpix(i,1)-P1(1,1) P1(3,2)*mpix(i,1)-P1(1,2) P1(3,3)*mpix(i,1)-P1(1,3);
            P1(3,1)*mpix(i,2)-P1(2,1) P1(3,2)*mpix(i,2)-P1(2,2) P1(3,3)*mpix(i,2)-P1(2,3);
            P2(3,1)*mpix(i,3)-P2(1,1) P2(3,2)*mpix(i,3)-P2(1,2) P2(3,3)*mpix(i,3)-P2(1,3);
            P2(3,1)*mpix(i,4)-P2(2,1) P2(3,2)*mpix(i,4)-P2(2,2) P2(3,3)*mpix(i,4)-P2(2,3)];
        B=[P1(1,4)-P1(3,4)*mpix(i,1);
            P1(2,4)-P1(3,4)*mpix(i,2);
            P2(1,4)-P2(3,4)*mpix(i,3);
            P2(2,4)-P2(3,4)*mpix(i,4)];
        reconstruction(:,i)=inv(A'*A)*A'*B;
        if size(colorp,3)==3
            color(i,1)=colorp(round(mpix(i,2)),round(mpix(i,1)),1);
            color(i,2)=colorp(round(mpix(i,2)),round(mpix(i,1)),2);
            color(i,3)=colorp(round(mpix(i,2)),round(mpix(i,1)),3);
        else
            color(i,1)=colorp(round(mpix(i,2)),round(mpix(i,1)),1);
            color(i,2)=colorp(round(mpix(i,2)),round(mpix(i,1)),1);
            color(i,3)=colorp(round(mpix(i,2)),round(mpix(i,1)),1);
        end
    end
    pc=pointCloud(single(reconstruction'),'Color',uint8(color));
end

function D=Decode(pc,k)
    B=pc>0;
    D=zeros(size(pc(:,:,1)));
    for i = 1:k 
        D = D + 2^(k-i)*(B(:,:,i));
    end
end

function D=GrayDecode(pc,k)
    B=pc(:,:,1);
    D=zeros(size(pc(:,:,1)));
    for i = 2:k 
        B(:,:,i) = xor(B(:,:,i-1),pc(:,:,i));
    end
    for i = 1:k 
        D = D + 2^(k-i)*(B(:,:,i));
    end
end

function [Graycodepattern,binarycodepattern]=Graycoding(height,width,angle,limit)
c=ceil(log(width)/log(2));
r=ceil(log(height)/log(2));
Col=0:2^c-1;	%設定垂直編碼值
Col=repmat(Col,[2^r,1]);	
% figure,imshow(Col/2^c)
Row=0:2^r-1;	%設定水平編碼值
Row=repmat((Row'),[1,2^c]);	
% figure,imshow(Row/2^r)
[m,n]=size(Col);
if c>limit
    c1=limit;
else
    c1=c;
end
if r>limit
    r1=limit;
else
    r1=r;
end
Col=imrotate(Col,angle,'crop');
Row=imrotate(Row,angle,'crop');
binary_col=zeros(m,n,c1);binary_row=zeros(m,n,r1);
for i=1:c1
    binary_col(:,:,i)=bitget(Col,c+1-i); 	%切割行位元平面
%     figure,imshow(binary_col(:,:,i))
end
for i=1:r1
    binary_row(:,:,i)=bitget(Row,r+1-i); 	%切割列位元平面
%     figure,imshow(binary_row(:,:,i))
end
Gray_col=zeros(m,n,c1);Gray_row=zeros(m,n,r1);
Gray_col(:,:,1) = binary_col(:,:,1);
Gray_row(:,:,1) = binary_row(:,:,1);
for i = 2:c1
   Gray_col(:,:,i) = xor(binary_col(:,:,i-1),binary_col(:,:,i)); 	%xor
%    figure,imshow(Gray_col(:,:,i))
end
for i=2:r1
    Gray_row(:,:,i) = xor(binary_row(:,:,i-1),binary_row(:,:,i)); 	%xor
%     figure,imshow(Gray_row(:,:,i))
end
c11=size(Gray_col,1)/2-height/2+1;
c12=size(Gray_col,1)/2+height/2;
c21=size(Gray_col,2)/2-width/2+1;
c22=size(Gray_col,2)/2+width/2;

binary_col=binary_col(c11:c12,c21:c22,:);
binary_row=binary_row(c11:c12,c21:c22,:);
Gray_col=Gray_col(c11:c12,c21:c22,:);
Gray_row=Gray_row(c11:c12,c21:c22,:);

binarycodepattern=cat(3,binary_col,binary_row); % concatenete array
Graycodepattern=cat(3,Gray_col,Gray_row);

end

function [HeightPatterns,WidthPatterns]=LargeGapGrayPatterns(height,width)
h1=ceil(log2(height));
w1=ceil(log2(width));

for i=1:max([h1,w1])
    transitions{i}=[];
end

transitions{2}=[0, 1, 0, 1];
transitions{3}=[0, 1, 0, 2, 0, 1, 0, 2];
transitions{4}=[0, 1, 2, 3, 2, 1, 0, 2, 0, 3, 0, 1, 3, 2, 3, 1];
transitions{5}=[0, 1, 2, 3, 4, 1, 2, 3, 0, 1, 4, 3, 2, 1, 4, 3, 0, 1, 2, 3, 4, 1, 2, 3, 0, 1, 4, 3, 2, 1, 4, 3];
transitions{6}=[0, 1, 2, 3, 4, 5, 0, 2, 4, 1, 3, 2, 0, 5, 4, 2, 3, 1, 4, 0, 2, 5, 3, 4, 2, 1, 0, 4, 3, 5, 2, 4, 0, 1, 2, 3, 4, 5, 0, 2, 4, 1, 3, 2, 0, 5, 4, 2, 3, 1, 4, 0, 2, 5, 3, 4, 2, 1, 0, 4, 3, 5, 2, 4];

if any([h1==10,w1==10])
    T0 = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
    T1 = to_4( 0, T0);T1=repmat(T1{4},[1,4]);
    T2 = to_4( 1, T1);T2=repmat(T2{1},[1,4]);
    T3 = to_4( 2, T2);T3=repmat(T3{2},[1,4]);
    transitions{10} = T_to_transition(T3);
end

for bits=2:max([h1,w1])
    if ~isempty(transitions{bits})
        if any([bits==h1,bits==w1])
            fprintf('gray code for %d bits has gap:%d\n',bits,gap(transitions{bits}));
        end
    else
%         fprintf('finding code for %d bits...\n',bits );
        i=(1:bits)';
        all_partitions=[i,bits-i];
        sum1=0;
        for i=1:size(all_partitions,1)
            if all_partitions(i,1)>=all_partitions(i,2) && all_partitions(i,2)>1
                sum1=sum1+1;
                partitions(sum1,:)=all_partitions(i,:);
            end
        end
        current_gap = 0;
        
        for i=1:size(partitions,1)
            P = interleave( transitions{partitions(i,1)}, transitions{partitions(i,2)});
            Z = P_to_transition(P,  transitions{partitions(i,1)}, transitions{partitions(i,2)});
            candidate_gap = gap(Z);
            if candidate_gap > current_gap
                current_gap = candidate_gap;
                transitions{bits} = Z;
            end
        end
        if any([bits==h1,bits==w1])
            if valid_circuit(transitions{bits})
                fprintf('gray code for %d bits has gap:%d\n',bits,gap(transitions{bits}));
            else
                fprintf('found in-valid gray code\n')
            end
        end
    end
end

hh1=zeros(h1,1);
th1=transitions{h1};
for i=1:length(th1)
    hh1(:,i+1)=hh1(:,i);
    hh1(th1(i)+1,i+1)=~hh1(th1(i)+1,i);
end

ww1=zeros(w1,1);
tw1=transitions{w1};
for i=1:length(tw1)
    ww1(:,i+1)=ww1(:,i);
    ww1(tw1(i)+1,i+1)=~ww1(tw1(i)+1,i);
end

for i=1:h1
    HeightPatterns(:,:,i)=repmat(hh1(i,1:end-1)',[1,2^w1]);
end
for i=1:w1
    WidthPatterns(:,:,i)=repmat(ww1(i,1:end-1),[2^h1,1]);
end
c11=size(HeightPatterns,1)/2-height/2+1;
c12=size(HeightPatterns,1)/2+height/2;
c21=size(WidthPatterns,2)/2-width/2+1;
c22=size(WidthPatterns,2)/2+width/2;

HeightPatterns=HeightPatterns(c11:c12,c21:c22,:);
WidthPatterns=WidthPatterns(c11:c12,c21:c22,:);
end

function Ans=interleave(A, B)
n=floor(log2(length(A)));
m=floor(log2(length(B)));
N=2^n;
M=2^m;

if N<M
    error('Error');
end

gap_A = gap(A);
gap_B = gap(B);

if gap_A<gap_B
    error('Error');
end

i=(1:2:M)';
st_pairs=[i,M-i];
[~,p]=sort(abs(st_pairs(:,2)./st_pairs(:,1)-gap_B/gap_A));
sorted_pairs=st_pairs(p,:);
best_pair = sorted_pairs(1,:);
s=best_pair(1);t=best_pair(2);
ratio=t/s;
P = 'b';

while length(P)<M
    b_to_a_ratio=sum(P=='b')/(sum(P=='a')+1);
    if b_to_a_ratio >= ratio
        P=[P,'a'];
    else
        P=[P,'b'];
    end
end
Ans=repmat(P,[1,N]);
end

function Z=P_to_transition(P, A, B)
Z = 0;
pos_a = 0;
pos_b = 0;
n=floor(log2(length(A)));
delta = n;
for p=1:length(P)
    if P(p)=='a'
        Z=[Z,A(mod(pos_a,length(A))+1)];
        pos_a=pos_a+1;
    else
        Z=[Z,B(mod(pos_b,length(B))+1)+delta];
        pos_b=pos_b+1;
    end
end
Z(1)=[];

end

function Ans=transition_to_code(transition_sequence)

code_sequence=0;
n=floor(log2(length(transition_sequence)));
code = 0;

for pos=1:length(transition_sequence)
    code=bitxor(code,bitshift(1,transition_sequence(pos)));
    code_sequence=[code_sequence,code];
end
Ans=code_sequence(1:end-1);

end

function return2=valid_circuit( transition_sequence )

return2=1;

n=floor(log2(length(transition_sequence)));

if length(transition_sequence)~=2^n
    fprintf('Length not valid\n')
    return2=0;
end
if ~all(transition_sequence<n)
    fprintf('Transition codes not valid\n')
    return2=0;
end
sorted_codes=sort(transition_to_code( transition_sequence ));

if ~all(sorted_codes==[0:1:2^n-1])
    fprintf('Not all Unique\n')
    return2=0;
end

end

function gap2=gap( transition_sequence )
a=transition_sequence ;
as_array=a;
gap2 = 1;
while gap2 < length(transition_sequence)
    if any(as_array==circshift(as_array,[1,gap2]))
        break;
    end
    gap2=gap2+1;
end

if gap2>=length(transition_sequence)
    gap2=0;
end

end

function permutations=to_4( i, sequence )
indices=0;

for j=1:length(sequence)
    x=sequence(j);
    if x==i
        indices=[indices,j];
    end
end
indices(1)=[];

for pos=1:length(indices)
    permutation=sequence;
    permutation(indices(pos))=4;
    permutations{pos}=permutation;
end

end

function transitions=T_to_transition(T)

T=T+1;
ticker=[0,0,0,0,0];
transitions=0;
for t=1:length(T)
    transistion = 2*T(t)+ 1*ticker(T(t));
    ticker(T(t))=~ticker(T(t));
    transitions=[transitions,transistion];
end
transitions(1)=[];
transitions=transitions-1-1;

end

function I = undistort_p(Idistorted, params)
fx=params.FocalLength(1);
fy=params.FocalLength(2);
cx=params.PrincipalPoint(1);
cy=params.PrincipalPoint(2);
k1=params.RadialDistortion(1);
k2=params.RadialDistortion(2);
k3=0;
p1=params.TangentialDistortion(1);
p2=params.TangentialDistortion(2);

K = [fx 0 cx; 0 fy cy; 0 0 1];

if size(Idistorted,2)==2
    Xp = inv(K)*[Idistorted(:,1),Idistorted(:,2),ones(size(Idistorted,1),1)]';
elseif size(Idistorted,1)==2
    Xp = inv(K)*[Idistorted(1,:);Idistorted(2,:);ones(1,size(Idistorted,1))];
end
r2 = Xp(1,:).^2+Xp(2,:).^2;
x = Xp(1,:);y = Xp(2,:);

x = x.*(1+k1*r2 + k2*r2.^2+k3*r2.^4) + 2*p1.*x.*y + p2*(r2 + 2*x.^2);
y = y.*(1+k1*r2 + k2*r2.^2+k3*r2.^4) + 2*p2.*x.*y + p1*(r2 + 2*y.^2);

% u and v are now the distorted cooridnates
u = fx*x + cx;
v = fy*y + cy;
I=[u;v]';

end