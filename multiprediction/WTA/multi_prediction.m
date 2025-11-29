clear;
fid2=fopen('t10k-labels.idx1-ubyte','r');
a=fread(fid2,2,'int32');
%label_esatta = fread(fid2,inf,'uchar'); 
label_esatta = fread(fid2,500,'uchar'); 
fclose(fid2);

for i = 0 : 9
    disp(['classify respect to model_' num2str(i) '... ']);
    eval(['!svm_classify -v 1 test500 model_' num2str(i) ' prediction_' num2str(i)]);
end
mat_pred = zeros(10,500);
for i = 0 : 9
    mat_pred(i+1,:) = load(['prediction_' num2str(i)]);
end
[vm, pred] = max(mat_pred);
pred = pred - 1; pred=pred(:);

diff = find(pred ~= label_esatta);
len_diff = length(diff);
n_test = 500;
fprintf('\n errori: %i su %i esempi \n',len_diff,n_test);

a = textread('test500','%s','delimiter','\n','bufsize',30000);
errori = zeros(len_diff,784);
for i = 1 : len_diff
    temp = a{diff(i)};
    j1=find(temp == ':');
    temp(j1)=' ';
    vet = str2num(temp);
    errori(i,vet(2:2:end)) = vet(3:2:end);
end

numero_fig = ceil(len_diff/16);
fig = 1;
for j = 1 : numero_fig
    figure; colormap(gray(256));
    for i = fig : min(fig+15,len_diff)
        subplot(4,4,i-fig+1);
        image( reshape( 256-errori(i,:),28,28)' );
        title( [ 'pred=' num2str(pred(diff(i))) '   ok=' num2str(label_esatta(diff(i))) ] );
        axis image
        axis off
    end
    fig = fig + 16;
end  