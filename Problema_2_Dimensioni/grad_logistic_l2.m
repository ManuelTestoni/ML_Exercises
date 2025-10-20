function [ g ] = grad_logistic_l2( z )
%
% Gradient of the w8a function; model LR   
%
global a b lambda
n = length(b);
nx = length(z);
temp = exp(-b.*(a*z));
temp2 = 1 + temp;
g = zeros(nx,1);
for i = 1 : n
    g = g + (-b(i)*temp(i))/temp2(i)*a(i,:)';
end
g = lambda*z + g/n;

