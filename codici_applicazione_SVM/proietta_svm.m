function px= proietta_svm(qk, y, e, l, u)
%
% Solves the problem    min  x'*x/2 - qk'*x
%                       subj to y'*x = e    
%                               l <= x <= u
% by algorithm [Pardalos-Kovoor; Math. Prog. 46 (1990) 321-328]
%
        iy = y; y = [];
        n = length(iy);
        qk = -qk;
        d = 0.5 * (iy' * qk + e);

        ind_1=find(iy==1);
        if length(ind_1)~=0
           a(ind_1)=(qk(ind_1)+l(ind_1))*0.5;
           b(ind_1)=(u(ind_1)+qk(ind_1))*0.5;
        end

        ind_1=find(iy~=1);
        if length(ind_1)~=0
           b(ind_1)=-(l(ind_1)+qk(ind_1))*0.5;
           a(ind_1)=-(u(ind_1)+qk(ind_1))*0.5;
        end


xmid  = 0;
xmin  = -1e300;
xmax  = 1e300;
uv    = 1:n;
xint  = [a b xmin xmax];
ts    = 0;
sw    = 0;
luv   = length(uv);
lxint = length(xint);
xint  = sort(xint);
aus   = 1;
ic    = 0;
rand('state', 300);

while aus
   ic = ic + 1;
   at = a(uv);
   bt = b(uv);

   xmold = xmid;
   xmid  = xint(ceil(lxint/2));
   if xmold == xmid
      xmid = xint(ceil(rand*lxint));
   end;

   s  = ts;
   s1 = sw;
   ind_1 = find(bt < xmid);
   if length(ind_1)~=0
     s  = s + sum(bt(ind_1));
   end 
   ind_1 = find(at > xmid);
   if length(ind_1)~=0
       s  = s + sum(at(ind_1));
   end    
   ind_1 = find(at <= xmid & bt >= xmid); 
   if length(ind_1)~=0
      s1 = s1 + length(ind_1);
   end
   
   testsum = s + s1*xmid;
   
   if testsum <= (d+(1e-15))
      xmin=xmid;
   end;
   if testsum >= (d-(1e-15))
      xmax=xmid;
   end;
   
   xint=xint( find(xint >= xmin  &  xint <= xmax) );
   lxint=length(xint);

   ind_1 = find(bt <= xmin);
   if length(ind_1)~=0
     ts = ts + sum(bt(ind_1));
   end  
   ind_1 = find(at >= xmax);
   if length(ind_1)~=0
     ts = ts + sum(at(ind_1));
   end  
   ind_1 = find(at <= xmin & bt >= xmax); 
   if length(ind_1)~=0
     sw = sw + length(ind_1);
   end  
   uv = uv(find( (at > xmin & at < xmax) | (bt > xmin & bt < xmax) ));
   luv=length(uv);
    
   if luv==0
      aus=0;
   end;
end;
% fine del ciclo while
if sw==0
   tt=xmin;
else
   tt= ( d-ts)/sw;
end;

% calcolo di x minimo per il problema tradotto con le regole di Kuhn-Tucker

for i=1:n
    if b(i)<=xmin
        y(i)=b(i);
    elseif a(i)>=xmax
        y(i)=a(i);
    elseif a(i)<=xmin & b(i)>=xmax
        y(i)=tt;
    else
        disp('problemi nel proiettore ');
    end
end

px = 2*y'.*iy - qk;

