function [x, vf, iter, info] = grad_pro_svm(Q, C, y, x, tol, maxit)
%
%               METODO DEL GRADIENTE PROIETTATO PER IL QP DELLE SVMs
%
%                               min 1/2 *x'*Q*x   -  x'*1
%                               s. to   0 <= x <= C,   y'*x = 0     
%  
% 
%
n=length(x);
x=x(:); y=y(:); l=zeros(n,1); u=C*ones(n,1); b=-ones(n,1);
x = proietta_svm(x, y, 0, l, u);
alpha_min = 1e-10; alpha_max = 1e10;
alpha = 1;
iter = 0;
info = 1;
grad = Q*x +b;
arresto = proietta_svm(x - grad,y,0,l,u) - x;

while  iter < maxit
    if norm(arresto,inf) > tol
        % STEP 2 (Proiezione)
        
        d = proietta_svm(x - alpha*grad,y,0,l,u) - x;
        
        % STEP 3 (Ricerca sulla direzione: regola di min. limitata)
        Qd = Q * d;
        if norm(Qd) > eps*norm(d)
            lam = min(-(grad' * d) / (d' * Qd), 1);
        else
            lam=1;
        end
        
        %lam =1;
        
        x = x + lam * d;
        grad = grad + lam * Qd;
        
        % STEP 4 (Aggiornamento di alpha)
        if norm(Qd) > eps*norm(d)
            if mod(iter,6) < 3
                alpha=(d' * Qd) / (Qd' * Qd);
            else
                alpha = (d' * d) / (d' * Qd);
            end
            alpha = max(alpha_min,min(alpha_max,alpha));
        else
            alpha = alpha_max;
        end
                
        arresto = proietta_svm(x - grad,y,0,l,u) - x;
        iter = iter + 1;
    else
        info = 0;
        break;
    end
end
vf = 0.5 * x' * ( grad + b );

