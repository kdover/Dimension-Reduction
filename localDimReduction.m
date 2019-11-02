function [ Y ] = localDimReduction(data, Trials, eta, K)
    dataSize = length(data(:,1));
    [Idx, D] = knnsearch(data,data,'K',K);
    %calc local dim, store n and t in 2 row vector
    localDim = returnLocalDists(D);
    %calc global dim
    %save as a matrix, each (i,j) entry is the squared
    %dist b/w ith and jth data point.
    origDist = calcDist(data);
    mu = mean(origDist,'all');
    sigma = std(origDist,0,'all');
    globalT = 2*mu/(sigma^2);
    globalN = 2*(mu^2)/(sigma^2);
    
    %initialize our data as normal, small variance
    Y = 0.001*randn(dataSize,2);
    trialNum = 0;
    while trialNum < Trials
        for i = 1:dataSize
            localN = localDim(1,i);
            localT = localDim(2,i);
            for j = 1:dataSize
                if ismember(j,Idx(i,:)) && j ~= i
                    %now calculate the difference
                    dif = Y(i,:)-Y(j,:);
                    actual_y_dist = norm(dif)^2;
                    highDist = localT*origDist(i,j);
                    %if dim < 3, just compare dist directly
                    if localN < 3
                        if actual_y_dist < highDist
                            Y(i,:)=Y(i,:)-eta*dif;
                        else
                            Y(i,:) = Y(i,:)+eta*dif;
                        end
                    else
                        xbound = localN-2;
                        expectedDist = -2*log(1-chi2cdf(highDist,localN));
                        ybound = -2*log(1-chi2cdf(xbound,localN));
                        if actual_y_dist > expectedDist && highDist < xbound
                            Y(i,:)=Y(i,:)-eta*dif;
                         elseif actual_y_dist < ybound && highDist > xbound
                             Y(i,:)=Y(i,:)+eta*actual_y_dist*dif;
                        elseif actual_y_dist < expectedDist && highDist < xbound
                            Y(i,:) = Y(i,:)+eta*dif;
                        end
                    end
                else
                    dif = Y(i,:)-Y(j,:);
                    actual_y_dist = norm(dif)^2;
                    highDist = globalT*origDist(i,j);
                    if globalN < 3
                        if actual_y_dist > highDist
                            Y(i,:)= Y(i,:)-eta*dif;
                        else
                            Y(i,:)= Y(i,:)+eta*dif;
                        end
                    else
                        expectedDist = -2*log(1-chi2cdf(highDist,globalN));
                        xbound = globalN-2;
                        ybound = -2*log(1-chi2cdf(xbound,globalN));
                        if actual_y_dst > expectedDist && highDist < xbound
                            Y(i,:) = Y(i,:)-eta*dif;
                         elseif actual_y_dist < ybound && highDist > xbound
                             Y(i,:) = Y(i,:)+eta*actual_y_dist*dif;
                        elseif actual_y_dist < expectedDist && highDist < xbound
                            Y(i,:) = Y(i,:)+eta*dif;
                        end
                    end
                end
            end
        end
        trialNum = trialNum+1;
    end
