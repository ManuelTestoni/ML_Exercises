clear; 
global a b lambda
dati = load('dati_dim2.txt');
b = dati(:,1);      % etichette degli esempi del training set
a = dati(:,[2 3]);  % gli esempi sono sulle riga della matrice a  
%%%%% grafico training set %%%%%%%%%%%
figure(1); clf;
plot(a(b>0,1),a(b>0,2),'or','LineWidth',2); hold on;
plot(a(b<0,1),a(b<0,2),'+b','LineWidth',2);
axis([0,1,0,1]);
%%%%%
n = length(b);      % numero di esempi
d = size(a,2);      % d = lunghezza esempi 
z0 = zeros(d,1);    % iterata iniziale per il metodo del gradiente  
toll = 1e-8;        % criterio di stopping: norm(g) < toll  
maxit = 2000;       % massimo numero di iterazioni
lambda = 1e-4;      % parametro di regolarizzazione 
tic;
[z_BB1,info_BB1,f_BB1,a_BB1] = BB1_solver('logistic_l2','grad_logistic_l2',z0,toll,maxit);
toc;
%%%%% grafico predittore 
plot([0 1],[0 -z_BB1(1)/z_BB1(2) ],'k');
legend('classe +1','classe -1','w^T x = 0');
title(['w = (',num2str([z_BB1(1) z_BB1(2)]),')']);
hold off;
%%%%% grafico discesa funzione obiettivo durante il processo di ottimizzazione 
figure(2); clf;
plot(f_BB1); xlabel('iterazioni'); ylabel('f(x_k)');