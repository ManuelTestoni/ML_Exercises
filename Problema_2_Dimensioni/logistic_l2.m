function [ f ] = logistic_l2( z )
%
%  w8a dataset; model LR 
%
global a b lambda 
temp2 = 1 + exp(-b.*(a*z));
f = 0.5*lambda*norm(z)^2 + mean( log(temp2) );
