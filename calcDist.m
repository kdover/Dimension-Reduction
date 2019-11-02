function [ B ] = calcDist(x)
size = length(x(:,1));
B = zeros(size,size);
for i = 1:size
    for j = (i+1):size
        diff = x(i,:)-x(j,:);
        dists = norm(diff)^2;
        B(i,j) = dists;
        B(j,i) = dists;
    end
end
