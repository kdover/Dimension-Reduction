function [ localDim ] = localDimReductionTest(data,target, Trials, eta, K,plot,error_size)
    dataSize = length(data(:,1));
    [Idx, D] = knnsearch(data,data,'K',K);
    %calc local dim, store n and t in 2 row vector
    localDim = returnLocalDim(D.^2);
    origDist = squareform(pdist(data)).^2;
    %expected_local_y_dist = returnAllDists(origDist,localDim);
    %find the global variables
    mu = mean(origDist,'all');
    sigma = std(origDist,0,'all');
    globalT = 2*mu/(sigma^2);
    globalN = 2*(mu^2)/(sigma^2);
    [expected_y_dist] = returnAllDists(origDist,localDim,Idx,globalT,globalN,error_size);
    %disp(globalN);
    %initialize our data as normal, small variance
    Y = 0.001*randn(dataSize,2);
    for k = 1:Trials
        for i = 1:dataSize
            for j = 1:dataSize
                dif = Y(i,:)-Y(j,:);
                actual_y_dist = norm(dif)^2;
                %if either are neighbors,just use local data
                if i~=j
                    Y(i,:) = Y(i,:)-eta*grad_function(actual_y_dist,expected_y_dist(i,j))*dif;
                end
            end
        end
        if plot == 1
           gscatter(Y(:,1),Y(:,2),target);
           title(k);
           drawnow
        end
    end