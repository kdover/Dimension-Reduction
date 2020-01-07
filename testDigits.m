rng(2);
justZerosAndOnes = digits(digits(:,65)< 4,:);
X = justZerosAndOnes(:,1:64);
target = justZerosAndOnes(:,65);
%178 for i=0, 182 for i = 1
K = 100;
% justZeros = digits(digits(:,65) == 0,:);
% [Idx, D] = knnsearch(justZeros,justZeros,'K',K);
% zero_n = -((1/(K-2)).*sum(log(D(1,2:K-1)./D(1,K)))).^(-1)
% justOnes = digits(digits(:,65)==1,:);
% [Idx, D] = knnsearch(justOnes,justOnes,'K',K);
% ones_n = -((1/(K-2)).*sum(log(D(1,2:K-1)./D(1,K)))).^(-1)
% bothClusters(1:178,:) = justZeros;
% bothClusters(179:360,:) = justOnes;
% bothClustersSquare = squareform(pdist(bothClusters)).^2;
% allDist = pdist(bothClusters).^2;
% zeroDist = pdist(justZeros).^2;
% onesDist = pdist(justOnes).^2;
% temp = zeros(1,20);
% index = 1;
% for i = 1:178
%     for j = 179:360
%        temp(index) = bothClustersSquare(i,j);
%        index = index+1;
%     end
% end
% zero_mu = mean(zeroDist,'all');
% ones_mu = mean(onesDist,'all');
% zero_sigma = std(zeroDist,0,'all');
% ones_sigma = std(onesDist,0,'all');
% %zero_t = 2*zero_mu/(zero_sigma^2);
% %ones_t = 2*ones_mu/(ones_sigma^2);
% zero_t = sqrt(2*zero_n/(zero_mu^2));
% ones_t = sqrt(2*ones_n/(ones_mu^2));
% %zero_n = 2*(zero_mu^2)/(zero_sigma^2);
% %ones_n = 2*(ones_mu^2)/(ones_sigma^2);
% scaled_t = sqrt(zero_t*ones_t);
% subplot(1,2,1)
% histogram(zero_t*zeroDist,'Normalization','probability');
% hold on
% histogram(zero_t*temp,'Normalization','probability');
% %histogram(scaled_t*allDist,'Normalization','probability');
% title('both scaled by local dim')
% hold off
% subplot(1,2,2)
% histogram(zero_t*zeroDist,'Normalization','probability');
% hold on
% histogram(scaled_t*temp,'Normalization','probability');
% %histogram(scaled_t*allDist,'Normalization','probability');
% title('scaled by average dim')
% hold off

plot = 1;
plotDist = 0;
error_size = 0;
Y = localDimReductionTest(X,target,50, 0.01, K,plot,error_size);
%Y = editCode(X,target,50, 0.01, K,plot,plotDist,error_size);



