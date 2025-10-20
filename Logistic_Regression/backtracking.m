function [alpha, info]=backtracking(fun,x,d,s,beta,sigma,FR,gf)
alpha = s;
while alpha > eps*norm(d)
    if FR - feval(fun,x+alpha*d) < -sigma * alpha  * gf
        alpha = alpha * beta;
    else
        info=0;
        return
    end
    
end
info=1;   
      
      