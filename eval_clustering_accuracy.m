function [r] = eval_clustering_accuracy(H, gnd, classnum, eval_num)

%   gnd: ground truth label (Nx1)

    accuracies = [];
    
    best_nmi = 0;
    best_idx = 0;    
    
    
    for i = 1 : eval_num
    
        label = litekmeans(H', classnum, 'Replicates', 20);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        labelgnmf = kmeans(H,classnum);
        result = ClusteringMeasure(gnd,labelgnmf);
        MIGNMF(i)=result(2);
            ACGNMF(i)=result(1);
            PUR(i)=result(3);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end 
    r=[];
    r.mi=mean(mean(MIGNMF));
    r.acc=mean(mean(ACGNMF));
    r.pur=mean(mean(PUR));
    r.st_mi=std2(MIGNMF);
    r.st_acc=std2(ACGNMF);
    r.st_pur=std2(PUR);
 
end

