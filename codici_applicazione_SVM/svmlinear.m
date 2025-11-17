clear all;
clc;
%  costruzione training set
% nome file dove caricare i dati
nomefile1 = input(' nome file classe  1 ','s');
nomefile2 = input(' nome file classe -1 ','s');
% carico i dati
tmp1 =  load(nomefile1);
[n1, m1] = size(tmp1);
tmp2 =  load(nomefile2);
[n2, m2] = size(tmp2);
if m1 ~= m2
    fprintf(' \n ERRORE: esempi con dimensione diversa \n');
    return
end
fprintf('\n lettura training set: %i esempi di classe 1,    %i esempi di classe -1 \n',n1,n2);
fprintf('\n  %%%%%%     ADDESTRAMENTO SVM     %%%%%% \n');
fprintf('\n Quanti esempi di classe  1 vuoi usare? (<=%i) ',n1); 
n1=input('');
fprintf(' Quanti esempi di classe -1 vuoi usare? (<=%i) ',n2); 
n2=input('');
% Associo le etichette
y = [ones(n1,1); -ones(n2,1)];

x = [tmp1(1:n1,:); tmp2(1:n2,:)];
%   costruzione iperpiano separatore ottimale generalizzato
%   mediante risoluzione del problema duale
%   Questo aiuta a gestire il trade off tra il learning del modello e
%   l'overfitting.
C = input('\n fornire il parametro di regolarizzazione C: ');
n = n1 + n2;
for i = 1 : n
    for j = i : n
        % Costruisco la matrice del problema
        % non funzionano così i software che risolvono SVM perchè in casi
        % reali questa matrice è troppo grande e non possiamo permetterci
        % di mantenerla in memoria.
        Q(i,j) = y(i) * y(j) * x(i,:) * x(j,:)';
    end
end
Q = triu(Q) + triu(Q,1)';
% Qua chiamo la funzione della proiezione del grandiente e poi cerco i
% vettori di supporto
[alpha, vf, iter, info] = grad_pro_svm(Q, C, y, zeros(n,1), 1e-6, 10000);
% Qui cercho gli alpha che sono diversi da zero, quindi maggiori della
% soglia che è indicata in qunato è quasi zero.
isv = find( (alpha > 1e-13)  );
% Considero alpha uguale a C quando è molto vicino a C.
ibsv = find( alpha >= C - 1e-13 );
isv_not_bound = find( ( alpha > 1e-13 ) & ( alpha < (C-1e-13) ) );
fprintf('\n  n. SV = %i,  n. BSV = %i\n',length(isv),length(ibsv));
%
%  equazione dell' iperpiano: w'*x +b = 0
%
w = sum( diag( alpha(isv) .* y(isv) ) * x(isv,:) )';
b = y(isv_not_bound(1)) - x(isv_not_bound(1),:)*w;
%
%  Classificazione di esempi: si prevede che nel file test dopo ogni
%  esempio ci sia la classe di appartenenza (1/-1)
%
ris = 's';
while ris=='s' | ris=='S'
     nometest = input('\n nome file test ','s');
     test = load(nometest);
     [nt, mt] = size(test);
     fprintf(' \n numero esempi test: %i \n',nt);
     if m1 ~= mt-1
        fprintf(' \n ERRORE: esempi test con dimensione diversa da esempi training  \n');
     else
        fprintf(' \n Quanti esempi vuoi testare ? (<=%i)  ',nt); ntest=input('');
        classetest = ( test(1:ntest,1:mt-1) * w + b);
        
        save svm_output2 classetest -ascii
        
        classetest = sign( classetest);
        
        iok = find(classetest == test(1:ntest,mt));
        inook = find(classetest ~= test(1:ntest,mt));
        fprintf('\n esempi test ben classificati: %i su %i \n',length(iok),ntest);
        clf;
        if length(inook)
            colormap(gray(256));
            notot=length(inook);
             for i=1:notot
                 subplot(ceil(notot/4),4,i);
                 image( reshape( 256 - test(inook(i),1:784),28,28)' );
                 axis off;
             end
        end
    end
    ris = input('\n altro test di classificazione? (s/n) ','s');
end
