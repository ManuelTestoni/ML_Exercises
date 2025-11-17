function [f, accuracy] = SG_LR(w_0, maxit, images, labels, testImages, testLabels, alpha,...
lambda)
accuracy = zeros(1, maxit+1);
f = zeros(1, maxit+1);
% Questo mi dice qual è il numero di element dentro
n = size(images,2);
w = w_0;
j =2;
% É il famosissimo w^Tx
b = w'*images;
a = b.*labels';
% Termine di regolazizazione, + contributo di ciascun elemento
f(1) = (1/2)*lambda*norm(w)^2 +(1/size(images,2))*sum(log(1+exp(-a)));
% Che cosa fa randperm? Facciamo la pescata casuale stocastica in diversi
% modi, andare a forzare una ciclicità di elementi è la cosa più
% convnienti. Quidni andiamo a vedere tutti gli elemeneti del dataset ma
% in ordine casuale, se volessi farlo in ordien veramente casuale dovrei
% utilizzare un altra funzione -> randi.
rr=randperm(n);
for i = 1:maxit*n
    if (mod(i, n) == 0)
        b = w'*images;
        a = b.*labels';
        f(j) = ((1/2)*lambda*norm(w)^2 +(1/size(images,2))*...
        sum(log(1+exp(-a))));
        % Calcolo dell'accuracy
        accuracy(j) = (size(testImages,2)-nnz(sign(w'*testImages) - testLabels'))/size(testImages,2);
        j= j+1;
        % In questo blocco entro ogni 11800 iterate.
        rr=randperm(n);
    end
    
    r=rr(mod(i-1,n)+1);
    g = (lambda*w-(labels(r,1)*exp(-w'*images(:,r)*labels(r,1)))/...
     (1+exp(-w'*images(:,r)*labels(r,1)))*images(:,r));

    w = w - alpha*g;

end