function [ dists ] = returnAllDists(origDist,localDim,Idx,globalT,globalN,error_size)
    dataSize = length(origDist(:,1));
    dists = zeros(dataSize,dataSize);
    for i = 1:dataSize
        for j = 1:i
            if ismember(j,Idx(i,:)) && ismember(i,Idx(j,:))
                dist1 = -2*log(chi2cdf(localDim(2,i)*origDist(i,j),localDim(1,i),'upper'));
                dist2 = -2*log(chi2cdf(localDim(2,j)*origDist(j,i),localDim(1,j),'upper'));
                dists(i,j) = max(dist1,dist2);
                dists(j,i) = dists(i,j);
            elseif globalN < 3
                dists(i,j) = globalT*origDist(i,j);
                dists(j,i) = dists(i,j);
            elseif globalN >= 3
                dists(i,j) = -2*log(chi2cdf(globalT*origDist(i,j),globalN,'upper'))+error_size;
                dists(j,i) = dists(i,j);
            end
        end
    end
end