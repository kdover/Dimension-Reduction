function [ c ] = grad_function(low_dim_dist,expected_dist)
    dif = low_dim_dist-expected_dist;
    %c = 2*sign(dif)*(dif).^2;
    if dif > 0
        c =2*(dif)^(-1)*exp(-0.2*(dif)^(-2));
    else
        c = sign(dif)*exp(-0.2*(dif)^(-2));
    end
    %c = 2*(dif).^(-3).*exp(-(dif).^(-2));
    %c = sign(dif)*sqrt(abs(dif))*exp(-(dif)^(-2));
    %c = sign(dif)*exp(-0.2*(dif)^(-2));
    %c = dif;
end
