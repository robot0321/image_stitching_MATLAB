function H = DLT(numDLT, numMatch, mp1, mp2)
    DLTrandIdx = randsample(numMatch, numDLT);
    selpts1 = mp1(DLTrandIdx,:);
    selpts2 = mp2(DLTrandIdx,:);
    
    A = [];
    for i=1:numDLT
        A = [A; [zeros(1,3),  -selpts1(i,:),  selpts2(i,2)*selpts1(i,:);
                 selpts1(i,:),   zeros(1,3), -selpts2(i,1)*selpts1(i,:)]];
    end

    % DLT1 ?
%     h = null(A);
%     [minh, idxh] = min(vecnorm(A*h, 2));
%     opth = h(:,idxh);

    % DLT2 ?
    [~,~,V] = svd(A);
    opth = V(:,end)/norm(V(:,end));
    minh = vecnorm(A*opth, 2);
    
    H = reshape(opth,3,3)';
end