function [f, accuracy] = mom_LR(w_0, maxit, images, labels, testImages, testLabels, alpha,...
lambda, beta)
f = zeros(1, maxit+1);
accuracy = zeros(1, maxit+1);
n = size(images,2);
m = size(images,1);
w = w_0;
j =2;
g_momentum = zeros(m,1);
b = w'*images;
a = b.*labels';
f(1) = (1/2)*lambda*norm(w)^2 +(1/size(images,2))*sum(log(1+exp(-a)));
rr=randperm(n);
for i = 1:maxit*n
    if (mod(i, n) == 0)
        b = w'*images;
        a = b.*labels';
        f(j) = (1/2)*lambda*norm(w)^2 +(1/size(images,2))*...
        sum(log(1+exp(-a))) ;
        accuracy(j) = (size(testImages,2)-nnz(sign(w'*testImages) - testLabels'))/size(testImages,2);
        j= j+1;
        rr=randperm(n);
    end
    r=rr(mod(i-1,n)+1);
    g = lambda*w-(labels(r,1)*exp(-w'*images(:,r)*labels(r,1)))/...
    (1+exp(-w'*images(:,r)*labels(r,1)))*images(:,r);
    g_momentum = beta*g_momentum + g;
    w = w - alpha*g_momentum;
end