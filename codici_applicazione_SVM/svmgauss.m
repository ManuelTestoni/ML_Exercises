clear all; clc;
%
%  costruzione training set
%
nomefile1 = input(' nome file classe  1 ','s');
nomefile2 = input(' nome file classe -1 ','s');
tmp1 =  load(nomefile1); 
[n1 m1] = size(tmp1);
tmp2 =  load(nomefile2); 
[n2 m2] = size(tmp2);
if m1 ~= m2
    fprintf(' \n ERRORE: esempi con dimensione diversa \n');
    return
end
fprintf(' \n lettura training set: %i esempi di classe 1,    %i esempi di classe -1 \n',n1,n2);
fprintf(' \n  %%%%%%     ADDESTRAMENTO SVM     %%%%%% \n');
fprintf(' \n  Quanti esempi di classe  1 vuoi usare? (<=%i)  ',n1); n1=input('');
fprintf(' \n  Quanti esempi di classe -1 vuoi usare? (<=%i) ',n2); n2=input('');
y = [ones(n1,1); -ones(n2,1)];
x = [tmp1(1:n1,:); tmp2(1:n2,:)];
%
%   costruzione iperpiano separatore ottimale generalizzato 
%   mediante risoluzione del problema duale
%
C = input(' fornire il parametro di regolarizzazione C: ');
sig = input(' fornire varianza: ');
sig = 2*sig^2;
n = n1 + n2;
% Quando calcolo gli elementi della matrice Q uso la gaussiana invece della
% formula di prima. Senza modificare nulla della struttura e della
% metodologia cambio la definizione del prodotto scalare e adatto ad un
% altro caso.
for i = 1 : n
    for j = i : n
Q(i,j) = y(i) * y(j) * exp( -( norm( x(i,:) - x(j,:) )^2 )/sig );
    end
end
Q = triu(Q) + triu(Q,1)'; 
%
% risoluzione QP con metodo del gradiente proiettato
%
[alpha, vf, iter, info] = grad_pro_svm(Q, C, y, zeros(n,1), 1e-3, 10000);
%    EVENTUALE USO DELLA QUADPROG
%options=optimset('LargeScale','off');
%[alphaqp,fval,exitflag]=quadprog(Q,-ones(n,1),[],[],y',0,zeros(n,1),C*ones(n,1),zeros(n,1),options);l
%disp('exitflag'); disp(exitflag);
%disp('err=');disp(norm(alpha-alphaqp));
%disp('valore funz.');disp([vf fval]);
isv = find( alpha > 1e-13 ) ;
ibsv = find( alpha >= (C - 1e-13) );
isv_not_bound = find( ( alpha > 1e-13 ) & ( alpha < (C-1e-13) ) );
fprintf('\n  n. SV = %i,  n. BSV = %i\n',length(isv),length(ibsv));
%
%  equazione dell' iperpiano: w'*x +b = 0
%
%w = sum( diag( alpha(isv) .* y(isv) ) * x(isv,:) )';
%b = y(isv(1)) - x(isv(1),:)*w;
b = - y(isv_not_bound(1))*(Q(isv_not_bound(1),:)*alpha -1);
%
%  Classificazione di esempi: si prevede che nel file test dopo ogni
%  esempio ci sia la classe di appartenenza (1/-1)
%
ris = 's';
while ris=='s' | ris=='S'
     nometest = input(' nome file test ','s');
     test = load(nometest);
     [nt mt] = size(test);
     fprintf(' \n numero esempi test: %i \n',nt);
     if m1 ~= mt-1
        fprintf(' \n ERRORE: esempi test con dimensione diversa da esempi training  \n');
     else
        fprintf(' \n  Quanti esempi vuoi testare ? (<=%i)  ',nt); ntest=input('');
        for i = 1:ntest
             for j = 1:length(isv) 
               ker(j)=exp( -norm( test(i,1:mt-1)- x(isv(j),:) )^2/sig ); 
             end  
             classetest_temp(i,1) = (ker*(alpha(isv).*y(isv))  + b);
             classetest(i) = sign(ker*(alpha(isv).*y(isv))  + b);
        end
        
        iok = find(classetest(:) == test(1:ntest,mt));
        inook = find(classetest(:) ~= test(1:ntest,mt));
        fprintf('\n esempi test ben classificati: %i su %i \n',length(iok),ntest);
        clf;
        if length(inook)
            colormap(gray(256));
            notot=length(inook);
             for i=1:notot
                 subplot(ceil(notot/4),4,i);
                 image( reshape( 255-test(inook(i),1:784),28,28)' );
                 axis image
                 axis off
             end
        end
    end
    ris = input(' altro test di classificazione? (s/n) ','s');
end