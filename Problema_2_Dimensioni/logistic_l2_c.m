function [ f ] = logistic_l2_c( z )
%
%f(w,c)=(1/N)sum( log(1+exp(-b_i*(a_i^Tw+c)))) + (lambda/2) norm([w;c])^2
%
global a b lambda 
w = z(1:end-1); c = z(end);
temp2 = 1 + exp(-b.*(a*w + c));
f = mean( log(temp2) ) + 0.5*lambda*norm(z)^2;
