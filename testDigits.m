rng(2);
justZerosAndOnes = digits(digits(:,65)< 4,:);
X = justZerosAndOnes(:,1:64);
target = justZerosAndOnes(:,65);
%178 for i=0, 182 for i = 1
K = 50;
choose_to_plot = 1;
plotDist = 0;
error_size = 30;
original_form = 0;
random_selection = 1;
weighted= 0;

%artificial data
% x1 = 0.000001*rand(150,5);
% x2 = 0.1*randn(150,5)+10;
% x3 = randn(150,5)-50;
% x4 = randn(150,5)+100;
x1 = randn(150,40)-1;
x2 = 2*randn(150,40);
x3 = randn(150,40)+1; 
x4 = randn(150,40)+5;
x = [x1;x2;x3;x4];
set_index = randperm(length(x(:,1)));

x = x(set_index,:);
targetx = [zeros(150,1); ones(150,1);(ones(150,1)+1);(ones(150,1)+2)];
targetx = targetx(set_index,:);




%Y = localDimReductionTest(X,target, 200, 0.01, K,choose_to_plot,error_size,original_form,random_selection,weighted);
Y = localDimReductionTest(x,targetx,50, 0.01, K,choose_to_plot,error_size,original_form,random_selection,weighted);
%Y = editCode(x,targetx,50, 0.01, K,1,error_size);



