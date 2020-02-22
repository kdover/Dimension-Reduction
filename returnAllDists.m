function [ dists ] = returnAllDists(origDist,localDim,Idx,globalT,globalN,error_size)
    [dataSize, kk] = size(Idx);
    dists = zeros(dataSize,dataSize);
    
    %% Find points which are within each others cluster 
    boo = zeros(dataSize,dataSize);
    boo(sub2ind([dataSize, dataSize], repmat(Idx(:,1), [1, kk-1]), Idx(:,2:end))) = 1;
    boo = boo|boo';
    
    %% local dists for all points
    localDist = zeros(dataSize, dataSize);
    tlocDist = repmat((localDim(2,:))', [1, dataSize]).*origDist;
    nlocDist = repmat((localDim(1,:))', [1, dataSize]);
    % chi2inv(1-chi2cdf(localDim(2,i)*origDist(i,j),localDim(1,i),'upper'),error_size)
    %P = max(P ./ sum(P(:)), realmin)
    %localDist = -2*log(chi2cdf(tlocDist,nlocDist,'upper'));
    localDist = chi2inv(1-chi2cdf(tlocDist,nlocDist,'upper'),error_size);
    localDist = min(localDist, localDist');
    localDist = min(localDist,max(localDist(~isinf(localDist))));
    localDist = localDist./max(localDist(~isinf(localDist)));
    %localDist = min(localDist,3*error_size);
    %localDist = localDist./sum(localDist,2);
    %localDist = localDist.*boo;
    %dists = localDist;

%     %% global dists for all points 
     globalDist = zeros(dataSize, dataSize);
     tGloDist = globalT.*origDist;
     %globalDist = -2*log(chi2cdf(tGloDist,globalN,'upper'));
     globalDist = chi2inv(1-chi2cdf(tGloDist,globalN,'upper'),4);
     %globalDist = min(globalDist,100);
     %globalDist = globalDist.*(1-boo);
% 
%     %% Masking 
     dists = localDist.*boo + 10.*globalDist.*(1-boo);
     %dists = localDist+globalDist;
      
%    w = bsxfun(@max,varN,varN');
