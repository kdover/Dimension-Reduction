function [ Y ] = localDimReductionTest(data,target, Trials, eta, K,choose_to_plot,error_size,original_form,random_selection,weighted)
    dataSize = length(data(:,1));
    %do nearest neighbors search
    [Idx, D] = knnsearch(data,data,'K',K);
    %calc local dim, store n and t in 2 row vector
    origDist = squareform(pdist(data)).^2;
    localDim = returnLocalDim(Idx,origDist,error_size);
    %expected_local_y_dist = returnAllDists(origDist,localDim);
    %find the global variables
    mu = mean(origDist,'all');
    sigma = std(origDist,0,'all');
    globalT = 2*mu/(sigma^2);
    globalN = 2*(mu^2)/(sigma^2);
    if ~weighted
        [expected_y_dist] = returnAllDists(origDist,localDim,Idx,globalT,globalN,error_size);
    end
    if weighted
        %[expected_weighted_y_dist] = returnAllDistsWeighted(origDist,localDim,Idx,globalT,globalN,error_size);
        %varN = 2.*localDim(1,:)./(localDim(2,:));
        %varN = varN./(sum(varN));
        %varN = 1- varN./(max(varN));
        varN = zeros(dataSize,1);
        for i = 1:dataSize
            varN(i,1) = std(localDim(1,Idx(i,:)));
        end
        varN = varN./(max(varN));
        varN = 1-1./(1+1000000000.^(-(varN-0.75)));
        %varN = ones(dataSize,1);
        [expected_weighted_y_dist] = returnAllDistsWeighted(origDist,localDim,Idx,globalT,globalN,error_size,1./(varN));
    end
    %keyboard
    %disp(globalN);
    %initialize our data as normal, small variance
    Y = 0.00001*randn(dataSize,2);
    broke = 0;
    index = 1;
    gscatter(Y(:,1),Y(:,2),target);
    drawnow
    while index <= Trials && broke == 0
        if original_form
            set_index1 = randperm(length(Y(:,1)));
            %set_index2 = randperm(length(Y(:,1)));
            for ii = 1:dataSize
                i = set_index1(ii);
                for jj = 1:dataSize
                    j = set_index1(jj);
                    dif = Y(i,:)-Y(j,:);
                    actual_y_dist = norm(dif)^2;
                    %if either are neighbors,just use local data
                    if i~=j
                        Y(i,:) = Y(i,:)-eta*grad_function(actual_y_dist,expected_y_dist(i,j))*dif;
                    end
                    if isnan(Y(i,1))
                        disp([index,i,j]);
                        disp(dif);
                        disp(actual_y_dist);
                        disp(expected_y_dist(i,j));
                        broke = 1;
                        break
                    end
                end
                if broke == 1
                    break
                end
            end
        end
        %instead, pull nearest neighbors first then take random sample to
        %push globally
        if random_selection
            set_index1 = randperm(length(Y(:,1)));
            for ii = 1:dataSize
                i = set_index1(ii);
                for jj = 1:K
                    j = Idx(i,jj);
                    dif = Y(i,:)-Y(j,:);
                    actual_y_dist = norm(dif)^2;
                    %if either are neighbors,just use local data
                    if i~=j
                        Y(i,:) = Y(i,:)-eta*grad_function(actual_y_dist,expected_y_dist(i,j))*dif;
                    end
                end
            end
            for ii = 1:dataSize
                i = set_index1(ii);
                global_index = randsample(dataSize,K);
                for jj = 1:length(global_index)
                    j = global_index(jj);
                    dif = Y(i,:)-Y(j,:);
                    actual_y_dist = norm(dif)^2;
                    %if either are neighbors,just use local data
                    if i~=j
                        Y(i,:) = Y(i,:)-eta*grad_function(actual_y_dist,expected_y_dist(i,j))*dif;
                    end
                end
            end
        end
        %if weighted, we use weights on the global and local distances
        if weighted
            set_index1 = randperm(length(Y(:,1)));
            %set_index2 = randperm(length(Y(:,1)));
            for ii = 1:dataSize
                i = set_index1(ii);
                for jj = 1:dataSize
                    j = set_index1(jj);
                    dif = Y(i,:)-Y(j,:);
                    actual_y_dist = norm(dif)^2;
                    %if either are neighbors,just use local data
                    if i~=j
                        Y(i,:) = Y(i,:)-eta*grad_function(actual_y_dist,expected_weighted_y_dist(i,j))*dif;
                    end
                end
            end
        end
        if choose_to_plot == 1
           gscatter(Y(:,1),Y(:,2),target);
           title(index);
           drawnow
        end
        index = index+1;
    end
