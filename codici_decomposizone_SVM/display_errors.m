clear; close all;
nomepred = input(' name of the prediction file: ','s');
nometest = input(' name of the test file: ','s');
pred = load(nomepred);
pred(find(pred>=0))=1; pred(find(pred<0))=-1;
a=textread(nometest,'%s','delimiter','\n','bufsize',30000);
for i=1:length(a)
    temp=a{i};   
    j1=find(temp==':');
    s = str2num(temp(1 : j1(1)-1 ));
    exact(i) = s(1);
end
pred=pred(:); exact=exact(:);
diverso = find(pred ~= exact);

notot=length(diverso);
if notot
  for i=1:notot
      temp=a{diverso(i)};
      j1=find(temp==':');
      temp(j1)=' ';
      vet = str2num(temp);
      testno(i,:) = zeros(1,784);
      testno(i,vet(2:2:end)) = vet(3:2:end);
      label_testno(i) = exact(diverso(i));
  end
  numero_fig = ceil(notot/16);
  fig = 1;
  for j = 1 : numero_fig
      figure; colormap(gray(256));
      for i = fig : min(fig+15,notot)
         subplot(4,4,i-fig+1);
         image( reshape( 255-testno(i,1:784),28,28)' );
         axis image
         axis off
         if label_testno(i)==1
             title('exact: 8');
         else
             title('exact: no 8');
         end
      end
      fig = fig + 16;
  end
end
