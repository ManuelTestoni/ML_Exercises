function [alpha_vect, f, accuracy] = adam_LR(w_0, maxit, images, labels, testImages, testLabels, alpha,...
lambda, beta_1, beta_2, epsilon)
f = zeros(1,maxit);
accuracy = zeros(1,maxit);
n = size(images,2);
m = size(images,1);
alpha_vect = zeros(1,maxit*n);
w = w_0;
g_mean = zeros(m,1);
g2_mean = zeros(m,1);
j =2;
b = w'*images;
a = b.*labels';
f(1) = (1/2)*lambda*norm(w)^2 +(1/size(images,2))*...
sum(log(1+exp(-a)));
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
%r = randi([1,size(images,2)],1,1);
g = lambda*w-(labels(r,1)*exp(-w'*images(:,r)*labels(r,1)))/...
 (1+exp(-w'*images(:,r)*labels(r,1)))*images(:,r);
g_mean = beta_1*g_mean + (1-beta_1)*g;
g2_mean = beta_2*g2_mean + (1-beta_2)*g.^2;
alpha1 = alpha*(sqrt(1- beta_2^i)/(1-beta_1^i));
w = w - alpha1*(g_mean./(sqrt(g2_mean)+epsilon*(sqrt(1-beta_2^i))));
alpha_vect(i) = alpha1;
end