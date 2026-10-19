function [U,V,core1,ob,iter]=cxgxabdntr(X,TY,core0,opt,A,W)%,D1)

eps=opt.eps;
Maxiter=opt.Maxiter;
lambda=opt.lambda;
alpha=opt.alpha;

n=size(X,2);
%通过LS计算LD和L
L=A-W;

sizecore0=size(core0,1);%将X分解为三阶张量的个数，这个数等于张量X的阶数，3
core1=core0;%{A0,B0,C0}={2*32*2;2*32*10;10*1440*2}
sg=cell(sizecore0,1);%用来存储每个因子张量的size,{2*32*2;2*32*10;10*1440*2}
for k1=1:sizecore0
    sg{k1,1}=size(core0{k1,1});
end
G22=cell(sizecore0,1);%用来存储将每个因子张量进行经典的mode-2展开,{32x4 ;32x20 ;1440x20 }
for k2=1:sizecore0
    G22{k2,1}=classical_mode_unfolding(core1{k2,1}, 2);
end
G20=G22;%mode-2展开

Xmod=cell(sizecore0,1);%用于存储将张量X进行不同方向展开的结果
for k3=1:sizecore0
    Xmod{k3,1}=mode_unfolding(TY,k3);
end
GK4=cell(sizecore0,1);%储存不等于d的核心张量的mode-2展开
GGK=cell(sizecore0,1);%储存不等于d的核心张量的mode-2展开的转置与自身相乘
obj0=0;
ob=zeros(Maxiter+1,1);
ob(1)=objapggsntr(TY,core1,L,lambda);%,alpha,DS);
for k=1:Maxiter
    for k4=1:sizecore0
    %%%%% X'=V*U'
        GK4{k4,1}=subchain_matrix(core1, k4);%Z^(!=d)_[2]
        if k4==sizecore0
            U=GK4{k4,1};
            UU=U'*U;
            V=G22{k4,1};
            VV=V'*V;
  
            V=V.*(Xmod{k4,1}*U+lambda*W*V)./max(V*UU+lambda*A*V,eps);
            U=U.*(Xmod{k4,1}'*V)./max(U*VV,0);

            G22{k4,1}=V;
        else
            GGK{k4,1}=GK4{k4,1}'*GK4{k4,1};
            G22{k4,1}=G22{k4,1}.*(Xmod{k4,1}*GK4{k4,1})./max(G22{k4,1}*GGK{k4,1},0);
            GK4{k4,1}=GK4{k4,1}.*(Xmod{k4,1}'*G22{k4,1})./max(GK4{k4,1}*G22{k4,1}'*G22{k4,1});
            G20{k4,1}=G22{k4,1};%保留上一次的G_(2)^n

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           
        end
        G22{k4,1}=max(0,G22{k4,1});
        core1{k4,1}=classical_mode_folding(G22{k4,1},2,sg{k4,1});

    end
    %停止条件
    obj1=objapggsntr(TY,core1,L,lambda);%,alpha,DS);
    ob(k+1)=obj1;
    err1=abs((obj1-obj0));
    disp(['the ', num2str(k), ' obj is ', num2str(ob(k)) , ' cost is ', num2str(err1)]);
    if err1/obj1<eps
        break;
    else
        obj0=obj1;
    end
end
iter=k;
end