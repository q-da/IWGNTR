function [feank,gndnk]=xuanPIE(fea,gnd,N,k,m)
%fea样本点 gnd标签信息 N总类数 k要选取的类数 m从一类中选取的个数

a=randperm(N,k);  %从0-9随机选取类别
row=m*k; 
gndnk=zeros(row,1);
% H=zeros(k,1);
d=zeros(N,1); 
ldx=zeros(N+1,1);
co=size(fea,2);
feank=zeros(m*k,co);
b=zeros(k,m);

for i=1:N
    [d(i),ldx(i)]=max(gnd==i);  %求各类的坐标 d只是判断储存 ldx储存各类的第一个坐标 例如ldx1(1)表示0这个类别出现的第一个坐标
end
ldx(N+1)=2857;  %70001可以换

% for i=1:k
%     H(i)=l(a(i)+2)-l(a(i)+1);
% end

H=ldx(a+1)-ldx(a); %选取的各类别含有的样本点个数
mh=floor(H/m);%求取多少数据来做平均


for i=1:k
    b(i,:)=sort(randperm(H(i),m));
    for j=1:m
        x=(ldx(a(i))-1)+b(i,j);%记录第一个下标
        y=(ldx(a(i))-1)+b(i,j)+mh(i);%记录第二个下标
        if y>size(fea,1)
            y=size(fea,1);
        end
        feank(m*(i-1)+j,:)=mean(fea(x:y,:),1);%这么多行取平均
        gndnk(m*(i-1)+j,1)=gnd(x,1);
    end
end



