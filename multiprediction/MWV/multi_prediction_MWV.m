clear;
fid2=fopen('t10k-labels.idx1-ubyte','r');
a=fread(fid2,2,'int32');
label_esatta = fread(fid2,inf,'uchar'); 
fclose(fid2);

%file_test = input(' dare nome file di test ','s');
file_test='test500';
a = textread(file_test,'%s','delimiter','\n','bufsize',30000);
n_test = length(a); 
label_esatta = label_esatta(1:n_test);

voting = zeros(10,n_test);

for cla_pos = 0 : 8
    for cla_neg = cla_pos + 1 : 9
        disp(['classifico rispetto model_' num2str(cla_pos) '_' num2str(cla_neg) '... ']);
        eval(['!svm_classify -v 1 ' file_test ' model_' num2str(cla_pos) '_' num2str(cla_neg) ...
              ' prediction_' num2str(cla_pos) '_' num2str(cla_neg) ]);
        temp = load(['prediction_' num2str(cla_pos) '_' num2str(cla_neg)]);  
        win = find(temp>0); no_win = setdiff(1:n_test,win);
        voting(cla_pos+1,win) = voting(cla_pos+1,win) + 1;
        voting(cla_neg+1,no_win) = voting(cla_neg+1,no_win) + 1;
        %pause
    end       
end
[vm, pred] = max(voting); pred = pred - 1; pred=pred(:);
diff = find(pred ~= label_esatta);  len_diff = length(diff);
fprintf('\n errori: %i su %i esempi \n',len_diff,n_test);

errori = zeros(len_diff,784);
for i = 1 : len_diff;
    temp = a{diff(i)};
    j1=find(temp == ':');   temp(j1)=' ';
    vet = str2num(temp);
    errori(i,vet(2:2:end)) = vet(3:2:end);
end

numero_fig = ceil(len_diff/16);    fig = 1;
for j = 1 : numero_fig
    figure; colormap(gray(256));
    for i = fig : min(fig+15,len_diff)
        subplot(4,4,i-fig+1);
        image( reshape( 256-errori(i,:),28,28)' );
        title( [ 'pred=' num2str(pred(diff(i))) '   ok=' num2str(label_esatta(diff(i))) ] );
        axis image;  axis off;
    end
    fig = fig + 16;
end  