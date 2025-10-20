function [x, info,f] = steepest(fun,grad,x,tol,maxit)
x = x(:);
g = feval(grad,x);
for i = 1 : maxit
   d = - g;
   FR = feval(fun,x); f(i)=FR;
   gf = g'*d;
   [alpha, inf] = backtracking(fun,x,d,100,0.5,1e-4,FR,gf);
   if inf>0
       fprintf(' \n problemi in armijo: alpha<=eps*s \n ');
   end    
   x = x + alpha * d;
   g = feval(grad,x);
   if norm(g) < tol
       info = 0;
       return
   end
end
info = 1;