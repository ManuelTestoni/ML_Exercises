close all; clear;
%%%%%%%%%%%%%%%     Test problems from Dai's paper   %%%%%%%%%%%%%%
% TP1 = w8a (model LR) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global a b lambda
load('w8a.mat'); %training data: examples in a, labels in b
%% 
b = b(:);        % column vector of labels
n = length(b);   % number of examples
d = size(a,2);   % d = length of the examples stored on the rows of a
x0 = zeros(d+1,1); % starting vector  
sty = {'g--', 'k--', 'r--', 'b', 'm','c','k:'};
toll = 1e-6;         % stopping criterion: norm(g) < toll  
maxit = 200;         % max number of iterations
lambda = 1e-4;
load('f_star_w8a_c'); % function evaluated in the exact solution of the problem with lambda=1e-4
%
%%%%%%%%%%%%%%   Barzilai-Borwein method %%%%%%%%%%%%%%
%
fprintf('\nOptimization solver: BB1\n\n');
tic;
[x_BB1_c, info_BB1,f_BB1,a_BB1] = BB1_solver('logistic_l2_c','grad_logistic_l2_c',x0,toll,maxit);
toc;
if info_BB1==0
    fprintf('iterations: %i\n',length(f_BB1)-1);
else
    fprintf('maxit reached\n');
end
save pesi_BB1_c x_BB1_c
%return 
figure(1); % F(x^*) is referred to the problem with lambda= 1e-4
leg = {'Barzilai-Borwein'}; 
semilogy(abs(f_BB1(1:end-1)-f_star_w8a_c),sty{2}); hold on; 
 xlabel('iterations');
 ylabel('F(x^{(k)}) - F(x^*)');
 title('Logistic Regression with $\ell_2$ regularization (w8a dataset)',...
     'FontSize',12,'Interpreter','Latex'); 
 %
%%%%%%%%%%%%%   steepest descent %%%%%%%%%%%%%%%%%%
%
fprintf('\nOptimization solver: steepest descent\n\n');
tic;
[x_s, info_s,f_s] = steepest('logistic_l2_c','grad_logistic_l2_c',x0,toll,maxit);
toc;
if info_s==0
    fprintf('iterations: %i\n',length(f_s)-1);
else
    fprintf('maxit reached\n');
end
save pesi_SD x_s
leg = [leg; {'Steepest'}]; 
semilogy(f_s(1:end-1)-f_star_w8a_c,sty{3});
legend(leg);