%% VALIDATE_RECURRENCES_MATLAB
% Independent formula-level checks for the distributed-delay chapter.
%
% Purpose:
%   Compare scalar Legendre-coefficient recurrences against direct numerical
%   evaluation of the defining integrals. The integral reference is coded
%   independently from the recurrence functions below.
%
% SCOPE AND REPRODUCIBILITY:
%   Record the MATLAB release and retain the command-window log when reproducing.
%   This driver validates selected coefficient formulas only; it does not execute
%   either full parameter-sweep script, assemble the full matrix P_n, or validate
%   a stability map. The main programs require separate runtime tests.
%
% Output files (written to the current folder):
%   matlab_constant_recurrence_checks.csv
%   matlab_exponential_recurrence_checks.csv
%   matlab_zero_mode_checks.csv

clear; clc;
familyNames = {'Gamma','GammaFlat','Lambda','barLambda','barLambdaFlat','barDelta','barDeltaFlat'};

%% Common setup
h = 0.1;
nConst = 5;
Nconst = 80;
zConstant = [0.05, 0.10, 0.25, 0.50, 1.00]; % z = h*lambda

constRows = cell(numel(zConstant), 7);
for iCase = 1:numel(zConstant)
    z = zConstant(iCase);
    lambda = z/h;
    [gDirect, lDirect, bgDirect] = constantDirect(h, lambda, nConst, Nconst);
    [gRec, lRec, bgRec] = constantRecurrence(h, lambda, nConst);

    [gAbs, gRel] = errorMeasures(gDirect, gRec);
    [lAbs, lRel] = errorMeasures(lDirect, lRec);
    [bgAbs, bgRel] = errorMeasures(bgDirect, bgRec);
    constRows(iCase,:) = {z, gAbs, gRel, lAbs, lRel, bgAbs, bgRel};
end
ConstantChecks = cell2table(constRows, 'VariableNames', ...
    {'h_lambda','Gamma_max_abs','Gamma_max_rel','Lambda_max_abs', ...
     'Lambda_max_rel','barGamma_max_abs','barGamma_max_rel'});
disp('Constant-kernel recurrence checks');
disp(ConstantChecks);
writetable(ConstantChecks, 'matlab_constant_recurrence_checks.csv');

%% Exact constant-kernel zero-mode values against direct quadrature
% The recurrence formulas contain 1/lambda and are not evaluated at lambda=0.
% Instead, compare the explicit limiting formulas with the defining integrals.
hZero = [0.1; 0.37];
zeroRows = cell(numel(hZero), 4);
for ih = 1:numel(hZero)
    hz = hZero(ih);
    [g0Direct, l0Direct, bg0Direct] = constantDirect(hz, 0, nConst, 40);
    [g0Exact, l0Exact, bg0Exact] = constantZeroModeExact(hz, nConst);
    [g0Abs, ~] = errorMeasures(g0Direct, g0Exact);
    [l0Abs, ~] = errorMeasures(l0Direct, l0Exact);
    [bg0Abs, ~] = errorMeasures(bg0Direct, bg0Exact);
    zeroRows(ih,:) = {hz, g0Abs, l0Abs, bg0Abs};
end
ZeroModeChecks = cell2table(zeroRows, 'VariableNames', ...
    {'h','Gamma0_max_abs','Lambda0_max_abs','barGamma0_max_abs'});
disp('Constant-kernel zero-mode checks');
disp(ZeroModeChecks);
writetable(ZeroModeChecks, 'matlab_zero_mode_checks.csv');

%% Exponential-kernel recurrence checks
% Avoid lambda=0 and gamma +/- lambda = 0 here because the inverse-based
% recurrence expressions are not defined at those values. The zero mode is a
% separate limiting case and must be treated with its explicit formulas.
gamma = -2.5;
nExp = 4;
Nexp = 16;
zExponential = [0.10, 0.30, 0.50, 1.00];
expRows = cell(numel(zExponential), 5);
for iCase = 1:numel(zExponential)
    z = zExponential(iCase);
    lambda = z/h;
    if abs(gamma-lambda) < 1e-12 || abs(gamma+lambda) < 1e-12
        error('A tested point lies on a singular denominator: gamma +/- lambda = 0.');
    end
    direct = exponentialDirect(h, lambda, gamma, nExp, Nexp);
    recur = exponentialRecurrence(h, lambda, gamma, nExp);
    famAbs = zeros(1,7); famRel = zeros(1,7);
    for f = 1:7
        [famAbs(f), famRel(f)] = errorMeasures(direct{f}, recur{f});
    end
    expRows(iCase,:) = {z, max(famAbs), max(famRel), ...
        strjoin(familyNames, ', '), familyNames{find(famRel==max(famRel),1,'first')}};
end
ExponentialChecks = cell2table(expRows, 'VariableNames', ...
    {'h_lambda','max_abs_error','max_rel_error','families_tested','worst_relative_family'});
disp('Exponential-kernel recurrence checks');
disp(ExponentialChecks);
writetable(ExponentialChecks, 'matlab_exponential_recurrence_checks.csv');

fprintf('\nFinished formula-level checks. These results do not validate the full MATLAB parameter-sweep scripts or P_n matrix assembly.\n');

%% Local functions
function [GammaD, LambdaD, BGammaD] = constantDirect(h, lambda, n, Nq)
    [t,w] = gaussLegendre01(Nq);
    P = legendreBasisTable(n,t);
    wt = w(:);

    GammaD = h * P * (wt .* exp(lambda*h*t(:)));

    T = repmat(t(:),1,Nq);
    U = T .* repmat(t(:).',Nq,1);
    W2 = wt * wt.';
    LambdaD = zeros(n,1);
    for k = 1:n
        integrand = W2 .* T .* repmat(P(k,:).',1,Nq) .* ...
            exp(lambda*h*(T-U));
        LambdaD(k) = h^2 * sum(integrand(:));
    end

    [T1,T2,Utri] = ndgrid(t,t,t);
    [W1,W2tri,Wu] = ndgrid(w,w,w);
    T2tri = T1 .* Utri;
    Wtri = W1 .* W2tri .* Wu .* T1;
    BGammaD = zeros(n,n);
    for j = 0:n-1
        Pj = shiftedLegendre(j,T1);
        for k = 0:n-1
            Pk = shiftedLegendre(k,T2tri);
            integrand = Wtri .* Pj .* Pk .* exp(lambda*h*(T1-T2tri));
            BGammaD(j+1,k+1) = h^2 * sum(integrand(:));
        end
    end
end

function [GammaR, LambdaR, BGR] = constantRecurrence(h, lambda, n)
    GammaR = zeros(n,1);
    GammaR(1) = (exp(h*lambda)-1)/lambda;
    if n > 1
        GammaR(2) = (exp(h*lambda)+1)/lambda - 2/(h*lambda)*GammaR(1);
    end
    for k = 2:n-1
        GammaR(k+1) = GammaR(k-1) - 2*(2*k-1)/(h*lambda)*GammaR(k);
    end

    LambdaR = zeros(n,1);
    LambdaR(1) = (GammaR(1)-h)/lambda;
    for k = 2:n
        LambdaR(k) = GammaR(k)/lambda;
    end

    BGR = zeros(n,n);
    BGR(1,1) = (GammaR(1)-h)/lambda;
    if n > 1
        BGR(1,2) = -GammaR(2)/lambda;
        BGR(2,1) = -BGR(1,2); % symmetry factor (-1)^(j+k) for j=0,k=1
        BGR(2,2) = ((2/(h*lambda)-1)*GammaR(2)-h/3)/lambda;
    end
    for j = 0:n-1
        for k = 0:n-1
            if k >= max(2,j)
                BGR(j+1,k+1) = BGR(j+1,k-1) + ...
                    2*(2*k-1)/(h*lambda)*BGR(j+1,k) ...
                    - h/((2*k+1)*lambda)*double(j==k) ...
                    + h/((2*k-3)*lambda)*double(j==k-2);
            elseif k < j
                BGR(j+1,k+1) = (-1)^(j+k)*BGR(k+1,j+1);
            end
        end
    end
end

function [GammaE, LambdaE, BGE] = constantZeroModeExact(h,n)
    GammaE = zeros(n,1);
    GammaE(1) = h;
    LambdaE = zeros(n,1);
    LambdaE(1) = h^2/2;
    if n > 1
        LambdaE(2) = h^2/6;
    end
    BGE = zeros(n,n);
    for j = 0:n-1
        for k = 0:n-1
            if k == 0
                BGE(j+1,k+1) = h^2/2*double(j==0) + h^2/6*double(j==1);
            else
                BGE(j+1,k+1) = h^2/(2*(2*k+1)) * ...
                    (double(j==k+1)-double(j==k-1))/(2*j+1);
            end
        end
    end
end

function direct = exponentialDirect(h, lambda, gamma, n, Nq)
    [t,w] = gaussLegendre01(Nq);
    P = legendreBasisTable(n,t);
    wt = w(:);
    GammaD = h * P * (wt .* exp(lambda*h*t(:)));
    GammaFlatD = h * P * (wt .* exp(gamma*h*(t(:)-1)));

    T = repmat(t(:),1,Nq);
    U = T .* repmat(t(:).',Nq,1);
    W2 = wt * wt.';
    LambdaD = zeros(n,1);
    for k = 1:n
        integrand = W2 .* T .* repmat(P(k,:).',1,Nq) .* ...
            exp(lambda*h*(T-U)+gamma*h*(U-1));
        LambdaD(k) = h^2 * sum(integrand(:));
    end

    [T1,T2,U3] = ndgrid(t,t,t);
    [W1,W2,W3] = ndgrid(w,w,w);
    W3all = W1 .* W2 .* W3;
    BL = zeros(n,n); BLflat = zeros(n,n);
    for j = 0:n-1
        Pj = shiftedLegendre(j,T1);
        for k = 0:n-1
            Pk = shiftedLegendre(k,T2);
            R = T1 .* U3;
            f = W3all .* T1 .* Pj .* Pk .* ...
                exp(lambda*h*R + gamma*h*(R-T1+T2-1));
            BL(j+1,k+1) = h^3 * sum(f(:));

            Rf = (T1-T2) .* U3;
            ff = -W3all .* (T1-T2) .* Pj .* Pk .* ...
                exp((gamma-lambda)*h*Rf + gamma*h*(-T1+T2-1));
            BLflat(j+1,k+1) = h^3 * sum(ff(:));
        end
    end

    [T1b,T2b,U1,U2] = ndgrid(t,t,t,t);
    [W1b,W2b,WU1,WU2] = ndgrid(w,w,w,w);
    W4 = W1b .* W2b .* WU1 .* WU2;
    BD = zeros(n,n); BDflat = zeros(n,n);
    for j = 0:n-1
        Pj = shiftedLegendre(j,T1b);
        for k = 0:n-1
            Pk = shiftedLegendre(k,T2b);
            R = T1b .* U1;
            Q = R .* U2;
            f = W4 .* (T1b.^2 .* U1) .* Pj .* Pk .* ...
                exp(gamma*h*(T1b+T2b-2-R-Q)+lambda*h*(R-Q));
            BD(j+1,k+1) = h^4 * sum(f(:));

            Rf = T1b .* U1;
            Qf = Rf + (T2b-Rf) .* U2;
            jac = T1b .* (T2b-Rf);
            ff = W4 .* jac .* Pj .* Pk .* ...
                exp(gamma*h*(T1b+T2b-2-Rf-Qf)+lambda*h*(Qf-Rf));
            BDflat(j+1,k+1) = h^4 * sum(ff(:));
        end
    end
    direct = {GammaD, GammaFlatD, LambdaD, BL, BLflat, BD, BDflat};
end

function recur = exponentialRecurrence(h, lambda, gamma, n)
    GammaR = zeros(n,1);
    GammaR(1) = (exp(h*lambda)-1)/lambda;
    if n > 1
        GammaR(2) = (exp(h*lambda)+1)/lambda - 2/(h*lambda)*GammaR(1);
    end
    for k = 2:n-1
        GammaR(k+1) = GammaR(k-1)-2*(2*k-1)/(h*lambda)*GammaR(k);
    end

    GammaFlatR = zeros(n,1);
    GammaFlatR(1) = (1-exp(-gamma*h))/gamma;
    if n > 1
        GammaFlatR(2) = 2*exp(-gamma*h)/gamma - ...
            (2/(h*gamma)-1)*GammaFlatR(1);
    end
    for k = 2:n-1
        GammaFlatR(k+1) = GammaFlatR(k-1)- ...
            2*(2*k-1)/(h*gamma)*GammaFlatR(k);
    end

    LambdaR = (GammaFlatR-exp(-gamma*h)*GammaR)/(gamma-lambda);
    BL = zeros(n,n); BLflat = zeros(n,n); BD = zeros(n,n); BDflat = zeros(n,n);
    for j = 0:n-1
        for k = 0:n-1
            BL(j+1,k+1) = (GammaR(j+1)*GammaFlatR(k+1) - ...
                (-1)^j*GammaFlatR(j+1)*GammaFlatR(k+1))/(gamma+lambda);
            BLflat(j+1,k+1) = (-1)^j * ...
                (GammaFlatR(j+1)*GammaFlatR(k+1) - ...
                GammaR(j+1)*GammaR(k+1)*exp(-(gamma+lambda)*h))/(gamma-lambda);
            BD(j+1,k+1) = ( ...
                (0.5*(-1)^j/gamma*GammaFlatR(j+1)*exp(-gamma*h) ...
                - GammaR(j+1)*exp(-gamma*h)/(gamma-lambda) ...
                + GammaFlatR(j+1)/(gamma-lambda) ...
                - 0.5*GammaFlatR(j+1)/gamma) * GammaFlatR(k+1))/(gamma+lambda);
            BDflat(j+1,k+1) = ( ...
                ((-1)^j/(gamma+lambda)*GammaR(j+1)*GammaR(k+1)*exp(-lambda*h)*exp(-2*gamma*h) ...
                - 1/(gamma+lambda)*GammaFlatR(j+1)*GammaR(k+1)*exp(-gamma*h) ...
                - 0.5*(-1)^j/gamma*GammaFlatR(j+1)*GammaFlatR(k+1)*exp(-gamma*h) ...
                + 0.5/gamma*GammaFlatR(j+1)*GammaFlatR(k+1)))/(gamma-lambda);
        end
    end
    recur = {GammaR, GammaFlatR, LambdaR, BL, BLflat, BD, BDflat};
end

function [absErr, relErr] = errorMeasures(reference, candidate)
    absErr = max(abs(reference(:)-candidate(:)));
    scale = max(abs(reference(:)));
    if scale == 0
        relErr = absErr;
    else
        relErr = absErr/scale;
    end
end

function P = legendreBasisTable(n,t)
    P = zeros(n,numel(t));
    for k = 0:n-1
        P(k+1,:) = shiftedLegendre(k,t(:)).';
    end
end

function y = shiftedLegendre(k,t)
    % ell_k(t) = P_k(2t-1), t in [0,1].
    x = 2*t-1;
    if k == 0
        y = ones(size(t));
    elseif k == 1
        y = x;
    else
        pkm2 = ones(size(t));
        pkm1 = x;
        for r = 2:k
            pk = ((2*r-1).*x.*pkm1-(r-1).*pkm2)/r;
            pkm2 = pkm1;
            pkm1 = pk;
        end
        y = pkm1;
    end
end

function [x,w] = gaussLegendre01(N)
    % Golub-Welsch construction of Gauss-Legendre nodes and weights on [0,1].
    i = (1:N-1).';
    beta = i ./ sqrt(4*i.^2-1);
    J = diag(beta,1)+diag(beta,-1);
    [V,D] = eig(J);
    [xRaw,order] = sort(diag(D));
    V = V(:,order);
    wRaw = 2*(V(1,:).^2).';
    x = (xRaw+1)/2;
    w = wRaw/2;
end

