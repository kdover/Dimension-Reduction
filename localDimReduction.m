function [ Y ] = localDimReduction(data,target, Trials, eta, K,plot)
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
        %disp(trialNum);
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
                        %if too big, push apart
                        if actual_y_dist > highDist
                            Y(i,:)=Y(i,:)-eta*dif;
                        else
                            %if too small, pull together
                            Y(i,:) = Y(i,:)+eta*dif;
                        end
                    else
                        xbound = localN-2;
                        expectedDist = -2*log(1-chi2cdf(highDist,localN));
                        ybound = -2*log(1-chi2cdf(xbound,localN));
                        %if y is much larger than x, pull together
                        if actual_y_dist > expectedDist && highDist < xbound
                            Y(i,:)=Y(i,:)-eta*dif;
                         %if y is much smaller than x, push apart
                        elseif actual_y_dist < ybound && highDist > xbound
                             Y(i,:)=Y(i,:)+eta*actual_y_dist*dif;
                        %if y is just smaller than x, push apart
                        elseif actual_y_dist < expectedDist && highDist < xbound
                            Y(i,:) = Y(i,:)+eta*dif;
                        end
                    end
                else
                    dif = Y(i,:)-Y(j,:);
                    actual_y_dist = norm(dif)^2;
                    highDist = globalT*origDist(i,j);
                    if globalN < 3
                        %if too large, pull together
                        if actual_y_dist > highDist
                            Y(i,:)= Y(i,:)-eta*dif;
                        %else, push apart
                        else
                            Y(i,:)= Y(i,:)+eta*dif;
                        end
                    else
                        expectedDist = -2*log(1-chi2cdf(highDist,globalN));
                        xbound = globalN-2;
                        ybound = -2*log(1-chi2cdf(xbound,globalN));
                        %if y is much larger than x, pull together
                        if actual_y_dist > expectedDist && highDist < xbound
                            Y(i,:) = Y(i,:)-eta*dif;
                        %if y is much smaller than x, push apart
                        elseif actual_y_dist < ybound && highDist > xbound
                             Y(i,:) = Y(i,:)+eta*actual_y_dist*dif;
                        %if y is just smaller than x, push apart
                        elseif actual_y_dist < expectedDist && highDist < xbound
                            Y(i,:) = Y(i,:)+eta*dif;
                        end
                    end
                end
            end
        end
        if plot == 1
           gscatter(Y(:,1),Y(:,2),target);
           drawnow
        end
        trialNum = trialNum+1;
    end
