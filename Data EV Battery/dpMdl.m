load data

M = 800; % number of energy samples
N = length(t);
delta = t(2)-t(1); % sample time
Eval = linspace(0,Emax,M);


%%%%%% compute the value function %%%%%
V = NaN*ones(N,M); % usage: V(time_index, energy_index)
% compute V_1(x_1)
for ii = 1:M
    x1 = Eval(ii);
    V(1,ii) = delta*F(x1/delta + Pl(1));
end
% compute V_k(x_k)
for k = 2:N
    for ii = 1:M % index of x_k
        xk = Eval(ii);
        vals = NaN*ones(M,1);
        for jj = 1:M % index of x_{k-1}
            xprev = Eval(jj); % This is x_{k-1}
            vals(jj) = V(k-1,jj) + delta*F((xk-xprev)/delta + Pl(k));
        end
        V(k,ii) = min(vals);
    end
end

%%%%%% compute optimal values %%%%%
xstar = NaN*ones(N,1);
% compute xstar(N)
[~,ii] =  min(V(N,:));
ii = ii(1);
xstar(N) = Eval(ii);
% compute xstar(k)
for k = (N-1):-1:1
    vals = NaN*ones(M,1);
    for jj = 1:M % index of x_k
        xk = Eval(jj);
        vals(jj) = V(k,jj) + delta*F((xstar(k+1) - xk)/delta + Pl(k+1));
    end
    [~,ii] =  min(vals);
    ii = ii(1);
    xstar(k) = Eval(ii);
end

ustar = NaN*ones(N,1);
ustar(1) = xstar(1)/delta;
for ii=2:N
    ustar(ii) = (xstar(ii) - xstar(ii-1))/delta;
end


Pg_dp = Pl+ustar;
Eg_dp = delta*cumsum(Pg_dp);

