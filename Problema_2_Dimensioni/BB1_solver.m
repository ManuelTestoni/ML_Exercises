function [x, info,f,a] = BB1_solver(fun,grad,x,tol,maxit)
x = x(:);
g = feval(grad,x);
rho = 1;
gold = g;
M = 5; %backtracking: lunghezza memoria per calcolo di f_ref
       % M=1 --> regola monotona di Armijo; M>1 --> regola non monotona 
Fold = -Inf*ones(M,1); %backtracking: vettore con valori di f(x_(k-j))
for i = 1 : maxit
   a(i) = rho;
   d = - rho*g;
   % in Fold(M-j) ci sono i valori f(x_(k-j)), j=0,1,...,M-1
   Fold(1:M-1) = Fold(2:M); Fold(M) = feval(fun,x); FR = max(Fold);
   f(i)=Fold(M);
   gf = g'*d;
   [alpha, inf] = backtracking(fun,x,d,1,0.5,1e-4,FR,gf);
   if inf>0
       fprintf(' \n problemi in backtraking: alpha<=eps*s \n ');
   end    
   s = alpha * d;
   x = x + s;
   g = feval(grad,x);
   y = g - gold; gold = g;
   % sogliatura del valore proposto da regole BB
   rho = max(1e-10, min(1e5,(s'*s)/(s'*y))); 
   if norm(g) < tol
       disp(['iter = ', num2str(i)])
       info = 0;
       return
   end
end
info = 1;