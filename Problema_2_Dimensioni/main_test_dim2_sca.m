clear; 
global a b lambda
a=load('dati_dim2.txt');
a1min= min(a(:,2)); a1max=max(a(:,2));
a2min= min(a(:,3)); a2max=max(a(:,3));
a(:,[2 3]) = a(:,[2 3])*diag( [2/(a1max-a1min) 2/(a2max-a2min) ]);
a(:,2)=a(:,2)-(a1max+a1min)/(a1max-a1min);
a(:,3)=a(:,3)-(a2max+a2min)/(a2max-a2min);
figure(10);
plot(a(a(:,1)>0,2),a(a(:,1)>0,3),'or','LineWidth',2); hold on;
plot(a(a(:,1)<0,2),a(a(:,1)<0,3),'+b','LineWidth',2);
axis([-1.5 1.5,-1.5,1.5]);
%%%%%
b = a(:,1);        % column vector of labels
a = a(:,[2 3]);
n = length(b);      % numero di esempi
d = size(a,2);      % d = lunghezza esempi 
z0 = zeros(d,1);    % iterata iniziale per il metodo del gradiente  
toll = 1e-4;        % criterio di stopping: norm(g) < toll  
maxit = 2000;       % massimo numero di iterazioni
lambda = 1e-4;      % parametro di regolarizzazione 
tic;
[z_BB1,info_BB1,f_BB1,a_BB1] = BB1_solver('logistic_l2','grad_logistic_l2',z0,toll,maxit);
toc;
%%%%% grafico predittore 
plot([-1 1],[ z_BB1(1)/z_BB1(2) -z_BB1(1)/z_BB1(2) ],'k');
legend('classe +1','classe -1','w^T x = 0');
title(['w = (',num2str([z_BB1(1) z_BB1(2)]),')']);
hold off;
%%%%% grafico discesa funzione obiettivo durante il processo di ottimizzazione 
figure(2); clf;
plot(f_BB1); xlabel('iterazioni'); ylabel('f(x_k)');