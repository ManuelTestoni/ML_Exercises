function [ g ] = grad_logistic_l2_c( z )
%
% Gradiente funzione obiettivo   
%
global a b lambda
w = z(1:end-1); c = z(end);
n = length(b);
nw = length(w);
temp = exp(-b.*(a*w + c));
temp2 = 1 + temp;
g = zeros(nw+1,1);
for i = 1 : n
    g(1:nw) = g(1:nw) + (-b(i)*temp(i))/temp2(i)*a(i,:)';
    g(nw+1) = g(nw+1) + (-b(i)*temp(i))/temp2(i);
end
g = lambda*z + g/n;

