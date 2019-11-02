function [ d ] = returnLocalDists( x )
    numDataPoints = length(x(:,1));
    localDim = zeros(2,numDataPoints);
    for i = 1:numDataPoints
        mu = mean(x(i,:));
        sigma = std(x(i,:));
        ourT = 2*mu/(sigma^2);
        n = 2*(mu^2)/(sigma^2);
        localDim(1,i) = n;
        localDim(2,i) = ourT;
    end
    d = localDim;
