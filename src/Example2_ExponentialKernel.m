%% PROGRAM: Legendre approximation-based stability test for distributed delay systems 
% CASE: Distributed delay with an exponential kernel.
% Author: Alejandro Castaño Hernández
% Advisors: Dr. Mathieu Bajodek / Dr. Sabine Mondié

%% Clean the workspace
clear, clc, hold on, format long
tic

% High-contrast, color-vision-conscious palette used consistently in all maps.
colorPositive = [0.000 0.447 0.698];       % blue (#0072B2)
colorNonPositive = [0.835 0.369 0.000];    % vermilion (#D55E00)
colorPending = [0.929 0.694 0.125];        % amber (#EDB120)
colorReview = [0.800 0.475 0.655];         % magenta/purple (#CC79A7)
colorDPartition = [0.25 0.25 0.25];       % dark gray for s=jw curve
colorDisagreement = [0.494 0.184 0.556];  % purple outline for diagnostic mismatch

% Parameter ranges and run mode.
% 'exploration' uses n=5 over a fine grid. 'criterion' computes a conservative
% prescribed-order bound and attempts that order only under the configured cap.
analysisMode = 'exploration'; % set to 'criterion' for prescribed-order runs
nExploration = 5;
maxOrderCriterion = 60; % matrix-valued coefficient arrays become expensive as n grows
explorationGridIntervals = 120;
criterionGridIntervals = 10; % coarser grid for the prescribed-order run
if strcmpi(analysisMode,'criterion')
    gridIntervals = criterionGridIntervals;
    modeTag = 'Criterion';
else
    gridIntervals = explorationGridIntervals;
    modeTag = 'Exploration';
end
% Every execution gets a unique output tag. Previous maps and CSVs are not
% deleted at startup, so an interrupted run cannot erase a previously completed result.
runTag = sprintf('%s_%s',modeTag,datestr(now,'yyyymmdd_HHMMSS'));
fprintf('Exponential-kernel run mode: %s; grid: %d x %d values; output tag: %s\n', ...
    analysisMode,gridIntervals+1,gridIntervals+1,runTag);
progressSummaryFile = sprintf('ScanProgress_ExponentialKernel_%s.csv',runTag);
progressMapFile = sprintf('StabilityMap_ExponentialKernel_%s_PROGRESS.png',runTag);
runtimeErrorFile = sprintf('RuntimeError_ExponentialKernel_%s.txt',runTag);

p1_min = -500; p1_max = 500;
p2_min = -500; p2_max = 500;

%Storage vectors
for STORAGE_VECTORS=1:1
    p1_vector = [];
    p2_vector = [];
    p1_nonpositive = [];
    p2_nonpositive = [];
    p1_evaluated = [];
    p2_evaluated = [];
    minEigPn_evaluated = [];
    rhoSym_evaluated = [];
    % Component-wise symmetry diagnostics. rhoDnIndependent is the residual obtained
    % by independently evaluating every D-block pair; rhoDnBlockStructured is the
    % residual of the block-symmetric D_n used in P_n. These are diagnostics,
    % not stability tests.
    componentSymmetryData = zeros(0,12); % [p1,p2,rhoU0,rhoTn,rhoSnSym,rhoDnIndependent,rhoDnStructured,rhoDndiagMax,rhoDnoffdiagMax,rhoWn,rhoBottom,rhoPn]
    rcondX_evaluated = [];
    minAbsHlambda_evaluated = [];
    lowSpectralScaleEvaluated = false(0,1);
    positiveDefinitePnEvaluated = false(0,1); % primary Cholesky test on block-symmetric P_n
    positiveDefiniteSymPartEvaluated = false(0,1); % diagnostic comparison using Sym(P_n)
    statusEvaluated = cell(0,1);
    numericalReviewPoints = zeros(0,10); % [p1,p2,rcondX,nUsed,nSafe,minAbsHlambda,minEigPn,rhoSym,Ubound,pdPn]
    numericalReviewReason = cell(0,1);
    orderUsedEvaluated = [];
    safeOrderEvaluated = [];
    logSafeOrderEvaluated = [];
    UboundEvaluated = [];
    orderPending = zeros(0,7); % [p1,p2,nSafe,logN,Ubound,mu,eta]
    orderPendingReason = cell(0,1);
    pointsPassingPrecomputationFilters = 0;
    pointsWithPnAssembled = 0;

    % Audit trail: rconds, spectral scale, symmetry residual, and reason.
    excluded_numeric = zeros(0,8); % [p1,p2,rcondL,rcond(gI-L),rcond(gI+L),rcondX,minAbsHlambda,rhoSym]
    exclusion_reason = cell(0,1);
end

% All computations in this script use double precision.

% The current order is selected pointwise after the semi-analytical boundary solve.
n  = nExploration;

p1_values = linspace(p1_min,p1_max,gridIntervals+1);
p2_values = linspace(p2_min,p2_max,gridIntervals+1);
for rowIndex = 1:numel(p1_values)
    p1 = p1_values(rowIndex);
    for p2 = p2_values
        try
        % System parameters:
        % x'(t) = A0x(t)+A1x(t-h)+B1*int_{-h}^{0} exp(\gamma \tau) x(t-\tau) d\tau
        h = 0.1;                   % Delay
        A0 = eye(2);               % Matrix A0
        A1 = -[0 2.5; 2.5 0];      % Matrix A1
        B1 =  [p1 0; 0 p2];        % Matrix B1
        g  = -2.5;                 % Parameter \gamma
        
        % ====================== DELAY LYAPUNOV MATRIX: EXPONENTIAL KERNEL ======================

        % Required variables
        A = g;
        B = B1;
        W = eye(size(A0,1));
        e0 = expm(A*0)*1;
        eh = expm(-A*h)*e0;
        
        % Initial arrays:
        nx  = size(A0,1);
        m   = size(A,1);
        In  = eye(nx);
        ni  = 1:nx;
        n2i = 1:nx^2;
        d   = (2 * m + 2)*nx^2;
        
        % Kronecker products
        xA0 = kron(A0',eye(nx)); 
        A0x = kron(eye(nx),A0'); 
        xA1 = kron(A1',eye(nx));
        A1x = kron(eye(nx),A1');
        xB  = zeros(nx*fliplr(size(B)));  %The fliplr command returns B with its columns reversed in the left-right direction.
        Bx  = zeros(size(xB));
        
        for i=0:m-1
            s = i * nx;
            t = s * nx;
            xB(:, t + n2i) = kron(B(s + ni,:)', eye(nx));
            Bx(:, t + n2i) = kron(eye(nx), B(s+ni,:)');
        end
        
        % Matrix L:
        L = [  xA0,                xA1,                 xB,                       zeros(nx^2, m * nx^2);
              -A1x,               -A0x,                 zeros(nx^2, m * nx^2),     -Bx;
              kron(e0, eye(nx^2)), -kron(eh, eye(nx^2)), -kron(A, eye(nx^2)),        zeros(m * nx^2, m * nx^2);
              kron(eh, eye(nx^2)), -kron(e0, eye(nx^2)),  zeros(m * nx^2, m * nx^2),  kron(A,eye(nx^2))];
        
        % ================= NUMERICAL SINGULARITY CHECK =================
        % The recurrences below require L^{-1} and (gI +/- L)^{-1}.
        % Skip this parameter point if a required matrix is singular or
        % numerically too ill-conditioned. Excluded points are NOT
        % classified as stable or unstable; they are recorded separately.
        IL = eye(size(L,1));
        tol_rcond = 1e-12;
        tolX = 1e-12;
        tolSym = 1e-8;
        tolRecurrence = 0.5; % Diagnostic only; the constant-kernel evidence does not justify a hard filter here.

        rcL     = rcond(L);
        rcMinus = rcond(g*IL - L);
        rcPlus  = rcond(g*IL + L);
        lambdaL = eig(L);
        minAbsHlambda = min(abs(h*lambdaL));

        if any(~isfinite([rcL, rcMinus, rcPlus, minAbsHlambda])) || ...
                any([rcL, rcMinus, rcPlus] < tol_rcond)
            excluded_numeric(end+1,:) = [p1,p2,rcL,rcMinus,rcPlus,NaN,minAbsHlambda,NaN];
            exclusion_reason{end+1,1} = 'L or a shifted matrix failed the rcond screen';
            continue;
        end
        % The 0.5 scale threshold was motivated mainly by constant-kernel tests.
        % For the exponential kernel it is only a diagnostic flag, not an exclusion:
        % the current tests do not justify using it as a hard admissibility condition.
        lowSpectralScaleWarning = minAbsHlambda < tolRecurrence;

        % Vector E:
        E = [zeros(nx^2), eye(nx^2), zeros(nx^2, m * nx^2), zeros(nx^2, m * nx^2) ];
        
        % Matrices I1, I2:
        I1 = [eye(m*nx^2), zeros(m*nx^2, m*d)] * expm(-[zeros(m*nx^2), -kron(eye(m), E); zeros(m*d, m*nx^2), kron(A, eye(d)) - kron(eye(m), L)]*h) * [zeros(m*nx^2, d); kron(e0, eye(d))];
        I2 = [eye(m*nx^2), zeros(m*nx^2, m*d)] * expm( [zeros(m*nx^2),  kron(eye(m), E); zeros(m*d, m*nx^2), kron(A, eye(d)) + kron(eye(m), L)]*h) * [zeros(m*nx^2, d); kron(eh, eye(d))];
        
        % Matrix M:
        M = [eye(nx^2), zeros(nx^2), zeros(nx^2, m*nx^2),   zeros(nx^2, m * nx^2);
            [zeros(m * nx^2, nx^2),  zeros(m * nx^2, nx^2), zeros(m * nx^2), eye(m * nx^2)]   - I1;
            [zeros(m * nx^2, nx^2),  zeros(m * nx^2, nx^2), eye(m * nx^2),   zeros(m * nx^2)] - I2;
            xA0,                   xA1,                 xB,             zeros(nx^2, m * nx^2)];
        
        % Matrix N:
        N = [zeros(nx^2),     -eye(nx^2),    zeros(nx^2, m * nx^2),     zeros(nx^2, m * nx^2);
             zeros(m * nx^2, d);
             zeros(m * nx^2, d);
            A1x,    A0x,   zeros(nx^2, m * nx^2),   Bx];
        
        % Matrix X and its conditioning check before solving the boundary system.
        X = M + N * expm(L*h);
        w = -[zeros((2 * m + 1 )*nx^2,1); W(:)];
        rcX = rcond(X);
        if ~isfinite(rcX) || (rcX < tolX && strcmpi(analysisMode,'criterion'))
            excluded_numeric(end+1,:) = [p1,p2,rcL,rcMinus,rcPlus,rcX,minAbsHlambda,NaN];
            exclusion_reason{end+1,1} = 'Boundary matrix X is singular/too ill-conditioned for criterion mode';
            continue;
        end
        boundaryConditionWarning = rcX < tolX;
        % In exploration mode, a finite but small rcond(X) is a warning, not
        % a hard filter. This recovers the broad P5 map while flagging points
        % whose semianalytical boundary solve may be inaccurate. Criterion mode
        % keeps the stricter condition because it aims at a theorem-level test.

        % Initial conditions of the semi-analytical method.
        pointsPassingPrecomputationFilters = pointsPassingPrecomputationFilters + 1;
        Uvec = X\w;
        if any(~isfinite(Uvec))
            excluded_numeric(end+1,:) = [p1,p2,rcL,rcMinus,rcPlus,rcX,minAbsHlambda,NaN];
            exclusion_reason{end+1,1} = 'Boundary solution Uvec contains nonfinite entries';
            continue;
        end

        % =====================  STABILITY TEST =====================

        % Auxiliary variables:
        E1 = [eye(nx^2), zeros(nx^2, (2 * m + 1)*nx^2)];
        U0 = reshape(E1*Uvec,nx,nx);

        if strcmpi(analysisMode,'criterion')
            % Conservative bound on ||U||_h. The semi-analytical solution is used
            % on [0,h], and the symmetry property covers the negative half interval.
            normG = norm(B1,2)*exp(max(0,-g*h));
            muSystem = norm(A0,2) + norm(A1,2) + h*normG;
            etaOrder = h*norm(A1,2) + h^2*normG;
            muLogL = max(real(eig((L+L')/2)));
            logUbound = log(max(norm(Uvec,2),realmin)) + max(0,h*muLogL);
            if isfinite(logUbound) && logUbound < log(realmax)-2
                Ubound = exp(logUbound);
            else
                Ubound = Inf;
            end
            [nSafe,orderInfo] = compute_legendre_order_bound(muSystem,etaOrder,Ubound,h,W);
            if ~isfinite(nSafe) || max(2,nSafe) > maxOrderCriterion
                orderPending(end+1,:) = [p1,p2,nSafe,orderInfo.logOrderBound,Ubound,muSystem,etaOrder];
                orderPendingReason{end+1,1} = sprintf('Conservative prescribed order unavailable or exceeds configured limit %d (%s)',maxOrderCriterion,orderInfo.status);
                continue;
            end
            n = max(2,nSafe); % the implemented recurrence initialization requires n>=2
        else
            % The order-bound calculation is unnecessary for the n=5 exploration
            % and would add substantial work at every grid point.
            n = nExploration;
            nSafe = NaN;
            Ubound = NaN;
            orderInfo = struct('status','not-computed-in-exploration','logOrderBound',NaN);
        end

        % Calculation of required inverses (after the conditioning check):
        Li = L\IL;
        gi = 1/g;
        gIL_Minus_L_in = (g*IL-L)\IL;
        gIL_Plus_L_in  = (g*IL+L)\IL;
        
        %% Qn and Rn terms:
      
        % Initialization for Gamma:
        Gamma    = cell(n,1);
        Gamma{1} = Li*( expm(h*L) - IL );                    %% Gamma[0]
        Gamma{2} = Li*( expm(h*L) + IL ) - 2/h*Li*Gamma{1};  %% Gamma[1]

        % Initialization for Gammab_k:
        Gammab    = cell(n,1);
        Gammab{1} = gi*IL*( IL - expm(-g*h*IL) );                        %% Gammab[0]
        Gammab{2} = 2*gi*IL*expm(-g*h*IL) - (2/h*gi*IL - IL)*Gammab{1};  %% Gammab[1]  

        % Initialization for Lambdak
        Lambda    = cell(n,1);
        Lambda{1} = gIL_Minus_L_in*(Gammab{1} - Gamma{1}*expm(-g*h*IL));   %% Lambda[0]
        Lambda{2} = gIL_Minus_L_in*(Gammab{2} - Gamma{2}*expm(-g*h*IL));   %% Lambda[1]

        % Recursive relations for Gamma, Lambdab and Lambda:
        for k=2:n-1
            Gamma{k+1}  = Gamma{k-1}  - 2*(2*k-1)/h*Li*Gamma{k};

            Gammab{k+1} = Gammab{k-1} - 2*(2*k-1)/h*gi*Gammab{k};
            Lambda{k+1} = gIL_Minus_L_in*( Gammab{k+1} - Gamma{k+1}*expm(-g*h*IL) );
        end

        % Construction of the terms Qn and Rn:
        Qn = zeros(nx,n*nx);
        Rn = zeros(nx,n*nx);

        for k=0:n-1
            Qk = reshape(E1*Gamma{k+1}*Uvec, nx, nx);
            Qn(:,k*nx+1:(k+1)*nx) = Qk'*A1;

            Rk = reshape(E1*Lambda{k+1}*Uvec, nx, nx);
            Rn(:,k*nx+1:(k+1)*nx) = Rk'*B1;
        end

        %% Term TnSym:
        % Initialization for GammaBar
        GammaBar = cell(n);
        GammaBar{1,1} =  Li*( Gamma{1} - h*IL );                        %% GammaBar[0,0]
        GammaBar{1,2} = -Li*Gamma{2};                                   %% GammaBar[0,1]
        GammaBar{2,2} =  Li*( ( 2/h*Li - IL )*Gamma{2} - h/3*IL );      %% GammaBar[1,1]

        % Recursive relations for GammaBar
        for j=1:n
            for k=1:n
                if k >= max(3,j)
                    GammaBar{j,k} = GammaBar{j,k-2} + 2*(2*k-3)/h*Li*GammaBar{j,k-1};
                    % Addition of Kronecker Delta terms:
                    if k == j
                        GammaBar{j,k} = GammaBar{j,k} - h/(2*j-1)*Li;
                    end
                    % Addition of Kronecker Delta terms:
                    if k == j+2
                        GammaBar{j,k} = GammaBar{j,k} + h/(2*j-1)*Li;
                    end
                else
                    if k < j
                        GammaBar{j,k} = (-1)^(j+k)*GammaBar{k,j};
                    end
                end
            end
        end
    
        %% Terms TnSym, Sn and Dn:
        TnSym = zeros(n*nx);
        Sn = zeros(n*nx);
        Dn = zeros(n*nx);
        % Diagnostic-only matrix: evaluates the closed-form D-block formula for
        % every (j,k), including lower-triangular pairs. In the theoretical D_n
        % matrix the lower triangle is specified by stars (transposes); see the
        % thesis, Chapter 6, Section 6.2.2, p. 88 and Eq. (6.15). Hence the matrix
        % used in P_n is constructed from upper-block formulas and their transposes.
        % The independent all-pairs matrix is saved only to diagnose redundant
        % lower-block evaluations; it is not used in the stability map.
        DnRawIndependent = zeros(n*nx);
        maxDiagonalDblockResidual = 0;
        maxOffDiagonalPairMismatch = 0;

        % Recursive relations and component-wise construction.
        for j = 0:n-1
            rj = j*nx+1:(j+1)*nx;
            for k = 0:n-1
                rk = k*nx+1:(k+1)*nx;
                Tjk = reshape(E1*GammaBar{j+1,k+1}*Uvec,nx,nx);
                TnSym(rj,rk) = A1'*(Tjk+(-1)^(j+k)*Tjk')*A1;

                Sjk  = reshape(E1*gIL_Plus_L_in*( Gamma{j+1} - (-1)^(j) * Gammab{j+1})*Gammab{k+1}*Uvec,nx,nx);
                SjkB = reshape(E1*gIL_Minus_L_in*(-1)^(j)*( Gammab{j+1}*Gammab{k+1} - Gamma{j+1}*Gamma{k+1}*expm(-h*(g*IL + L) ) ) *Uvec,nx,nx);
                Sn(rj,rk) = A1'*(Sjk + SjkB')*B1;

                % Evaluate every index pair for a diagnostic comparison, but
                % assemble the actual Dn from its upper triangle and set the
                % lower block by transpose, as prescribed by the block matrix.
                Djk  = reshape( E1*gIL_Plus_L_in*(  1/2*(-1)^(j)*gi*Gammab{j+1}*exp(-g*h) - gIL_Minus_L_in*Gamma{j+1}*exp(-g*h) + gIL_Minus_L_in*Gammab{j+1} - 1/2*gi*Gammab{j+1}   ) * Gammab{k+1}*Uvec,nx,nx );
                DjkB = reshape( E1*gIL_Minus_L_in*( gIL_Plus_L_in*(-1)^(j)*Gamma{j+1}*Gamma{k+1}*expm(-h*L)*exp(-2*h*g) - gIL_Plus_L_in*Gammab{j+1}*Gamma{k+1}*exp(-g*h) - 1/2*gi*(-1)^(j)*Gammab{j+1}*Gammab{k+1}*exp(-g*h) + 1/2*gi*Gammab{j+1}*Gammab{k+1} )*Uvec,nx,nx);
                Dblock = B1'*(Djk+DjkB')*B1;
                DnRawIndependent(rj,rk) = Dblock;
                if j == k
                    rhoDdiag = norm(Dblock-Dblock','fro')/max(1,norm(Dblock,'fro'));
                    maxDiagonalDblockResidual = max(maxDiagonalDblockResidual,rhoDdiag);
                end
                if k >= j
                    Dn(rj,rk) = Dblock;
                    if k > j
                        Dn(rk,rj) = Dblock';
                    end
                end
            end
        end
        for jj = 0:n-1
            rrj = jj*nx+1:(jj+1)*nx;
            for kk = jj+1:n-1
                rrk = kk*nx+1:(kk+1)*nx;
                Dup = DnRawIndependent(rrj,rrk);
                Dlow = DnRawIndependent(rrk,rrj);
                rhoPair = norm(Dup-Dlow','fro')/max(1,norm(Dup,'fro'));
                maxOffDiagonalPairMismatch = max(maxOffDiagonalPairMismatch,rhoPair);
            end
        end
        rhoDnIndependent = norm(DnRawIndependent-DnRawIndependent','fro') / ...
            max(1,norm(DnRawIndependent,'fro'));
        % rhoDn measures the residual of the block-symmetric matrix actually
        % specified by the criterion. rhoDnIndependent is a separate diagnostic.

        %% Term IIn:
        IIn = 1/h*kron(diag(2*(0:n-1)+1),eye(nx));

        % Matrix P_n: retain the contributions and assemble D_n according to
        % its block-symmetric definition; independently evaluated lower blocks are diagnostic only.
        Wn = IIn\eye(size(IIn));
        SnSym = Sn + Sn';
        bottomBlock = TnSym + SnSym + Dn + Wn;
        Pn = [U0 Qn+Rn; Qn'+Rn' bottomBlock];
        pointsWithPnAssembled = pointsWithPnAssembled + 1;

        % Nonfinite P_n entries are a genuine numerical failure. A finite but
        % noticeably nonsymmetric P_n is retained as a review point rather than
        % silently accepted or discarded from the parameter map.
        if any(~isfinite(Pn(:)))
            excluded_numeric(end+1,:) = [p1,p2,rcL,rcMinus,rcPlus,rcX,minAbsHlambda,NaN];
            exclusion_reason{end+1,1} = 'P_n contains nonfinite entries';
            continue;
        end
        rhoSym = norm(Pn-Pn','fro')/max(1,norm(Pn,'fro'));
        % Relative symmetry residuals for every independently assembled block.
        rhoU0 = norm(U0-U0','fro')/max(1,norm(U0,'fro'));
        rhoTn = norm(TnSym-TnSym','fro')/max(1,norm(TnSym,'fro'));
        rhoSnSym = norm(SnSym-SnSym','fro')/max(1,norm(SnSym,'fro'));
        rhoDn = norm(Dn-Dn','fro')/max(1,norm(Dn,'fro'));
        rhoWn = norm(Wn-Wn','fro')/max(1,norm(Wn,'fro'));
        rhoBottom = norm(bottomBlock-bottomBlock','fro')/max(1,norm(bottomBlock,'fro'));
        if ~isfinite(rhoSym)
            excluded_numeric(end+1,:) = [p1,p2,rcL,rcMinus,rcPlus,rcX,minAbsHlambda,rhoSym];
            exclusion_reason{end+1,1} = 'P_n symmetry diagnostic is nonfinite';
            continue;
        end
        Pn_sym = (Pn+Pn')/2;
        [~, Px] = chol(Pn_sym);
        % The block structure above is symmetric by construction, so the primary
        % positivity test is applied to Pn itself. The symmetric-part Cholesky is
        % retained only as a diagnostic comparison.
        [~, PxDirect] = chol(Pn);
        minEigPn = min(real(eig(Pn)));
        p1_evaluated(end+1) = p1;
        p2_evaluated(end+1) = p2;
        minEigPn_evaluated(end+1) = minEigPn;
        rhoSym_evaluated(end+1) = rhoSym;
        componentSymmetryData(end+1,:) = [p1,p2,rhoU0,rhoTn,rhoSnSym,rhoDnIndependent,rhoDn,maxDiagonalDblockResidual,maxOffDiagonalPairMismatch,rhoWn,rhoBottom,rhoSym];
        minAbsHlambda_evaluated(end+1) = minAbsHlambda;
        lowSpectralScaleEvaluated(end+1,1) = lowSpectralScaleWarning;
        positiveDefinitePnEvaluated(end+1,1) = (PxDirect == 0); % primary test on structured P_n
        positiveDefiniteSymPartEvaluated(end+1,1) = (Px == 0); % diagnostic on Sym(P_n)
        orderUsedEvaluated(end+1) = n;
        safeOrderEvaluated(end+1) = nSafe;
        logSafeOrderEvaluated(end+1) = orderInfo.logOrderBound;
        UboundEvaluated(end+1) = Ubound;

        rcondX_evaluated(end+1,1) = rcX;
        symmetryWarning = rhoSym > tolSym;
        reviewFlag = symmetryWarning || lowSpectralScaleWarning || boundaryConditionWarning;
        if reviewFlag
            numericalReviewPoints(end+1,:) = [p1,p2,rcX,n,nSafe,minAbsHlambda,minEigPn,rhoSym,Ubound,double(PxDirect==0)];
            reasons = {};
            if symmetryWarning
                reasons{end+1} = 'P_n symmetry residual';
            end
            if lowSpectralScaleWarning
                reasons{end+1} = 'low spectral scale';
            end
            if boundaryConditionWarning
                reasons{end+1} = 'small rcond(X)';
            end
            numericalReviewReason{end+1,1} = ['Review required: ' strjoin(reasons,', ')];
        end

        if strcmpi(analysisMode,'exploration')
            % Preserve the finite-order exploratory map. Since P_n is assembled
            % block-symmetrically, classify using chol(P_n); warnings remain metadata.
            if PxDirect == 0
                p1_vector(end+1) = p1;
                p2_vector(end+1) = p2;
                if reviewFlag
                    statusEvaluated{end+1,1} = 'positive-definite-at-n5-review-required';
                else
                    statusEvaluated{end+1,1} = 'positive-definite-at-n5';
                end
            else
                p1_nonpositive(end+1) = p1;
                p2_nonpositive(end+1) = p2;
                if reviewFlag
                    statusEvaluated{end+1,1} = 'not-positive-definite-at-n5-review-required';
                else
                    statusEvaluated{end+1,1} = 'not-positive-definite-at-n5';
                end
            end
        elseif reviewFlag
            statusEvaluated{end+1,1} = 'criterion-order-pending-numerical-review';
        elseif PxDirect == 0
            p1_vector(end+1) = p1;
            p2_vector(end+1) = p2;
            statusEvaluated{end+1,1} = 'positive-definite-at-prescribed-order';
        else
            p1_nonpositive(end+1) = p1;
            p2_nonpositive(end+1) = p2;
            statusEvaluated{end+1,1} = 'not-positive-definite-at-prescribed-order';
        end

        catch ME
            % An unexpected implementation error must not be disguised as an
            % excluded or unstable parameter point. Preserve the checkpoint,
            % write the MATLAB stack/parameter coordinates, and stop for diagnosis.
            fidError = fopen(runtimeErrorFile,'a');
            if fidError >= 0
                fprintf(fidError,'Unexpected error at (p1,p2)=(%.16g,%.16g)\n',p1,p2);
                fprintf(fidError,'Identifier: %s\nMessage: %s\n',ME.identifier,ME.message);
                fprintf(fidError,'%s\n\n',ME.getReport('extended','hyperlinks','off'));
                fclose(fidError);
            end
            fprintf(2,'Unexpected error at (p1,p2)=(%.16g,%.16g) [%s]: %s\n', ...
                p1,p2,ME.identifier,ME.message);
            rethrow(ME);
        end
    end

    if rowIndex==1 || mod(rowIndex,10)==0 || rowIndex==numel(p1_values)
        fprintf('Progress exponential kernel: row %d/%d; P_n tests=%d; PD=%d; non-PD=%d; exclusions=%d\n', ...
            rowIndex,numel(p1_values),numel(p1_evaluated),numel(p1_vector),numel(p1_nonpositive),size(excluded_numeric,1));
        progressTable = table(rowIndex,numel(p1_values),numel(p1_evaluated),numel(p1_vector), ...
            numel(p1_nonpositive),size(excluded_numeric,1), ...
            'VariableNames',{'completedRows','totalRows','pointsWithPnDiagnostic','positiveDefiniteSymPart', ...
            'notPositiveDefiniteSymPart','excludedPoints'});
        writetable(progressTable,progressSummaryFile);
        fProgress = figure('Visible','off','Color','w');
        axProgress = axes(fProgress); hold(axProgress,'on');
        if ~isempty(p1_vector), plot(axProgress,p1_vector,p2_vector,'.','Color',colorPositive,'MarkerSize',7); end
        if ~isempty(p1_nonpositive), plot(axProgress,p1_nonpositive,p2_nonpositive,'.','Color',colorNonPositive,'MarkerSize',6); end
        if ~isempty(excluded_numeric), plot(axProgress,excluded_numeric(:,1),excluded_numeric(:,2),'kx','MarkerSize',3); end
        if ~isempty(numericalReviewPoints), plot(axProgress,numericalReviewPoints(:,1),numericalReviewPoints(:,2),'d','Color',colorReview,'MarkerFaceColor','none','MarkerSize',4); end
        axis(axProgress,[p1_min p1_max p2_min p2_max]); axis(axProgress,'square'); grid(axProgress,'on'); box(axProgress,'on');
        xlabel(axProgress,'p_1'); ylabel(axProgress,'p_2');
        title(axProgress,sprintf('Exponential kernel: partial n=%d scan, rows %d/%d',nExploration,rowIndex,numel(p1_values)),'Interpreter','none');
        print(fProgress,progressMapFile,'-dpng','-r150'); close(fProgress);
    end
end
elapsed_time = toc;
totalGridPoints = numel(p1_values)*numel(p2_values);
fprintf('\nElapsed time: %.4f s\n', elapsed_time);
fprintf('Total grid points: %d\n', totalGridPoints);
fprintf('Points reaching the P_n test: %d\n', numel(p1_evaluated));
if strcmpi(analysisMode,'criterion')
    fprintf('Analysis mode: prescribed-order criterion (conservative order bound).\n');
    fprintf('Positive definite at tested conservative order: %d\n', numel(p1_vector));
    fprintf('Not positive definite at tested conservative order: %d\n', numel(p1_nonpositive));
else
    fprintf('Analysis mode: exploratory finite-order screen, n=%d.\n', nExploration);
    fprintf('Positive definite at exploration order: %d\n', numel(p1_vector));
    fprintf('Not positive definite at exploration order: %d\n', numel(p1_nonpositive));
end
fprintf('Numerical exclusions before/at P_n: %d\n', size(excluded_numeric,1));
fprintf('Points pending because prescribed order is too large/unavailable: %d\n', size(orderPending,1));
fprintf('Points flagged for symmetry, spectral-scale and/or boundary-solve review: %d\n', size(numericalReviewPoints,1));
fprintf('Empirical spectral-scale warning threshold: min |h*eig(L)| < %.3g; warning only, not a hard exclusion.\n', tolRecurrence);

% Save a compact summary alongside the point-wise audit tables.
summaryFile = sprintf('Summary_ExponentialKernel_%s.csv',runTag);
summaryTable = table({analysisMode}, gridIntervals+1, gridIntervals+1, totalGridPoints, ...
    pointsPassingPrecomputationFilters, pointsWithPnAssembled, numel(p1_evaluated), ...
    sum(positiveDefinitePnEvaluated), sum(~positiveDefinitePnEvaluated), ...
    sum(positiveDefiniteSymPartEvaluated), ...
    sum(positiveDefinitePnEvaluated(:) ~= positiveDefiniteSymPartEvaluated(:)), ...
    size(excluded_numeric,1), size(numericalReviewPoints,1), size(orderPending,1), ...
    'VariableNames', {'analysisMode','gridP1Count','gridP2Count','totalGridPoints', ...
    'pointsPassingPrecomputationFilters','pointsWithPnAssembled','pointsWithPnDiagnostic', ...
    'positiveDefinitePn','notPositiveDefinitePn','positiveDefiniteSymmetricPartDiagnostic', ...
    'PnVsSymmetricPartDisagreement','numericallyExcluded','flaggedForNumericalReview','pendingOrder'});
writetable(summaryTable,summaryFile);

% Stability map: same classification colors as the constant-kernel map.
figMap = figure('Color','w','Units','centimeters','Position',[2 2 20 19], ...
    'PaperPositionMode','auto');
axMap = axes(figMap); hold(axMap,'on');
if strcmpi(analysisMode,'criterion')
    plotLabel = 'P_n positive definite at prescribed order';
    plotTitle = 'Exponential kernel: prescribed-order criterion scan';
else
    plotLabel = sprintf('P_%d positive definite (exploratory screen)',nExploration);
    plotTitle = sprintf('Exponential kernel: order-%d exploratory stability screen',nExploration);
end
if ~isempty(p1_vector)
    plot(p1_vector,p2_vector,'.','Color',colorPositive, ...
        'MarkerSize',8,'DisplayName',plotLabel);
end
if ~isempty(p1_nonpositive)
    plot(p1_nonpositive,p2_nonpositive,'.','Color',colorNonPositive, ...
        'MarkerSize',7,'DisplayName','P_n not positive definite at tested order');
end
if ~isempty(excluded_numeric)
    plot(excluded_numeric(:,1),excluded_numeric(:,2),'kx', ...
        'MarkerSize',4,'DisplayName','Excluded by numerical/admissibility filters');
end
if ~isempty(orderPending)
    plot(orderPending(:,1),orderPending(:,2),'s','Color',colorPending, ...
        'MarkerSize',4,'DisplayName','Prescribed order pending');
end
% Numerical-review markers are deliberately NOT overlaid on this figure:
% the uploaded run showed that nearly every point carries a review flag,
% which completely obscures the exploratory blue/orange classification. The
% full review list remains in its CSV and a separate residual map is saved below.
legacyDisagreement = positiveDefiniteSymPartEvaluated(:) ~= positiveDefinitePnEvaluated(:);
if any(legacyDisagreement)
    plot(p1_evaluated(legacyDisagreement),p2_evaluated(legacyDisagreement),'o', ...
        'Color',colorDisagreement,'MarkerFaceColor','none','MarkerSize',4, ...
        'DisplayName','Direct vs symmetric-part Cholesky differ');
end
axis(axMap,[p1_min p1_max p2_min p2_max]); axis(axMap,'square');
grid(axMap,'on'); box(axMap,'on');
xlabel(axMap,'p_1'); ylabel(axMap,'p_2');
numSymWarnings = sum(rhoSym_evaluated > tolSym);
title(axMap,{plotTitle, sprintf('%d/%d evaluated points exceed symmetry tolerance; see residual map', ...
    numSymWarnings,numel(rhoSym_evaluated))},'Interpreter','none');
applyExponentialFigureStyle(figMap,axMap,2);

excludedFile = sprintf('ExcludedPoints_ExponentialKernel_%s.csv',runTag);
if ~isempty(excluded_numeric)
    excludedTable = array2table(excluded_numeric, ...
        'VariableNames', {'p1','p2','rcond_L','rcond_gI_minus_L','rcond_gI_plus_L','rcond_X','minAbsHlambda','rhoSym'});
    excludedTable.Reason = exclusion_reason;
else
    excludedTable = table('Size',[0 9], ...
        'VariableTypes',{'double','double','double','double','double','double','double','double','string'}, ...
        'VariableNames',{'p1','p2','rcond_L','rcond_gI_minus_L','rcond_gI_plus_L','rcond_X','minAbsHlambda','rhoSym','Reason'});
end
writetable(excludedTable,excludedFile);
evaluatedFile = sprintf('EvaluatedPoints_ExponentialKernel_%s.csv',runTag);
if ~isempty(p1_evaluated)
    nEvaluated = numel(p1_evaluated);
    assert(numel(p2_evaluated)==nEvaluated && numel(orderUsedEvaluated)==nEvaluated && ...
        numel(safeOrderEvaluated)==nEvaluated && numel(logSafeOrderEvaluated)==nEvaluated && ...
        numel(UboundEvaluated)==nEvaluated && numel(rcondX_evaluated)==nEvaluated && ...
        numel(minAbsHlambda_evaluated)==nEvaluated && numel(lowSpectralScaleEvaluated)==nEvaluated && ...
        numel(minEigPn_evaluated)==nEvaluated && numel(rhoSym_evaluated)==nEvaluated && ...
        numel(positiveDefinitePnEvaluated)==nEvaluated && ...
        numel(positiveDefiniteSymPartEvaluated)==nEvaluated && numel(statusEvaluated)==nEvaluated, ...
        'ExponentialKernel:EvaluatedColumnLengthMismatch', ...
        'Evaluated-point diagnostic columns do not have consistent lengths.');
    evaluatedTable = table(repmat({analysisMode},nEvaluated,1), ...
        p1_evaluated(:),p2_evaluated(:),orderUsedEvaluated(:),safeOrderEvaluated(:), ...
        logSafeOrderEvaluated(:),UboundEvaluated(:),rcondX_evaluated(:),minAbsHlambda_evaluated(:), ...
        lowSpectralScaleEvaluated(:),minEigPn_evaluated(:),rhoSym_evaluated(:), ...
        positiveDefinitePnEvaluated(:),positiveDefiniteSymPartEvaluated(:),statusEvaluated(:), 'VariableNames', ...
        {'analysisMode','p1','p2','orderUsed','conservativeOrderBound','logOrderBound', ...
        'U_norm_upper_bound','rcondX','minAbsHlambda','lowSpectralScaleWarning','minEigPn', ...
        'rhoSym','positiveDefinitePn','positiveDefiniteSymmetricPartDiagnostic','status'});
else
    evaluatedTable = table('Size',[0 15], ...
        'VariableTypes',{'string','double','double','double','double','double','double','double','double','logical','double','double','logical','logical','string'}, ...
        'VariableNames',{'analysisMode','p1','p2','orderUsed','conservativeOrderBound','logOrderBound', ...
        'U_norm_upper_bound','rcondX','minAbsHlambda','lowSpectralScaleWarning','minEigPn', ...
        'rhoSym','positiveDefinitePn','positiveDefiniteSymmetricPartDiagnostic','status'});
end
writetable(evaluatedTable,evaluatedFile);
% Save component-wise diagnostics. rhoDnIndependent measures disagreement when the
% formula is independently evaluated for every (j,k); rhoDnBlockStructured is the residual
% after constructing D_n from upper blocks and their prescribed transposes. The former
% is retained as a diagnostic, not used alone to reject the structurally assembled P_n.
componentFile = sprintf('ComponentSymmetryDiagnostics_ExponentialKernel_%s.csv',runTag);
componentTable = array2table(componentSymmetryData, 'VariableNames', ...
    {'p1','p2','rhoU0','rhoTn','rhoSnSym','rhoDnIndependent','rhoDnBlockStructured','maxDiagonalDblockResidual','maxOffDiagonalPairMismatch','rhoWn','rhoBottomBlock','rhoPn'});
writetable(componentTable,componentFile);
reviewFile = sprintf('NumericalReviewPoints_ExponentialKernel_%s.csv',runTag);
if ~isempty(numericalReviewPoints)
    numericalReviewTable = array2table(numericalReviewPoints,'VariableNames', ...
        {'p1','p2','rcondX','orderUsed','conservativeOrderBound','minAbsHlambda','minEigPn','rhoSym','U_norm_upper_bound','positiveDefinitePn'});
    numericalReviewTable.Reason = numericalReviewReason;
else
    numericalReviewTable = table('Size',[0 11], ...
        'VariableTypes',{'double','double','double','double','double','double','double','double','double','double','string'}, ...
        'VariableNames',{'p1','p2','rcondX','orderUsed','conservativeOrderBound','minAbsHlambda','minEigPn','rhoSym','U_norm_upper_bound','positiveDefinitePn','Reason'});
end
writetable(numericalReviewTable,reviewFile);

pendingFile = sprintf('PendingCriterionOrder_ExponentialKernel_%s.csv',runTag);
if ~isempty(orderPending)
    pendingOrderTable = array2table(orderPending,'VariableNames', ...
        {'p1','p2','conservativeOrderBound','logOrderBound','U_norm_upper_bound','muSystem','eta'});
    pendingOrderTable.Reason = orderPendingReason;
else
    pendingOrderTable = table('Size',[0 8], ...
        'VariableTypes',{'double','double','double','double','double','double','double','string'}, ...
        'VariableNames',{'p1','p2','conservativeOrderBound','logOrderBound','U_norm_upper_bound','muSystem','eta','Reason'});
end
writetable(pendingOrderTable,pendingFile);

% No D-partition overlay is added: the available constant-kernel overlay belongs to a different parameterization.

% Save the classification map and a separate symmetry-residual map. The
% former remains legible even if every grid point requires numerical review.
mapFile = sprintf('StabilityMap_ExponentialKernel_%s.png',runTag);
print(figMap,mapFile,'-dpng','-r300');
fprintf('Saved parameter map: %s\n',mapFile);

symmetryMapFile = sprintf('SymmetryResidualMap_ExponentialKernel_%s.png',runTag);
figSym = figure('Color','w','Units','centimeters','Position',[2 2 20 19], ...
    'PaperPositionMode','auto');
axSym = axes(figSym);
logRho = log10(max(rhoSym_evaluated,realmin));
scatter(axSym,p1_evaluated,p2_evaluated,20,logRho,'filled');
colormap(figSym,parula(256));
axis(axSym,[p1_min p1_max p2_min p2_max]); axis(axSym,'square');
grid(axSym,'on'); box(axSym,'on');
xlabel(axSym,'p_1'); ylabel(axSym,'p_2');
title(axSym,sprintf('Exponential kernel: log_{10} symmetry residual of P_{%d}',nExploration), ...
    'Interpreter','tex');
cb = colorbar(axSym); ylabel(cb,'log_{10}(\rho_{sym})','Interpreter','tex');
applyExponentialFigureStyle(figSym,axSym,0);
set(cb,'FontName','Arial','FontSize',9,'LineWidth',0.8);
set(get(cb,'Label'),'FontName','Arial','FontSize',10);
print(figSym,symmetryMapFile,'-dpng','-r300');
close(figSym);
fprintf('Saved symmetry residual map: %s\n',symmetryMapFile);

% Separate diagnostic: discrepancy between independently evaluating all D-block
% pairs and constructing D_n from the upper triangle plus prescribed transposes.
% The component CSV additionally reports diagonal-block symmetry and the maximum
% off-diagonal pair mismatch; these distinguish a diagonal formula issue from a
% lower-triangle reconstruction issue.
dnMismatchMapFile = sprintf('DnIndependentBlockMismatchMap_ExponentialKernel_%s.png',runTag);
figDnMismatch = figure('Color','w','Units','centimeters','Position',[2 2 20 19], ...
    'PaperPositionMode','auto');
axDn = axes(figDnMismatch);
valsDn = componentSymmetryData(:,6); % rhoDnIndependent
finiteDn = isfinite(valsDn);
if any(finiteDn)
    scatter(axDn,componentSymmetryData(finiteDn,1),componentSymmetryData(finiteDn,2), ...
        18,log10(max(valsDn(finiteDn),realmin)),'filled');
    colormap(figDnMismatch,parula(256));
end
axis(axDn,[p1_min p1_max p2_min p2_max]); axis(axDn,'square');
grid(axDn,'on'); box(axDn,'on');
xlabel(axDn,'p_1'); ylabel(axDn,'p_2');
title(axDn,'Exponential kernel: independently evaluated lower-block mismatch', ...
    'Interpreter','none');
cbDn = colorbar(axDn);
ylabel(cbDn,'log_{10}(relative residual)','Interpreter','none');
applyExponentialFigureStyle(figDnMismatch,axDn,0);
set(cbDn,'FontName','Arial','FontSize',9,'LineWidth',0.8);
set(get(cbDn,'Label'),'FontName','Arial','FontSize',10);
print(figDnMismatch,dnMismatchMapFile,'-dpng','-r300');
close(figDnMismatch);
fprintf('Saved independent D-block mismatch map: %s\n',dnMismatchMapFile);

if exist(progressMapFile,'file'), delete(progressMapFile); end
if exist(progressSummaryFile,'file'), delete(progressSummaryFile); end


function applyExponentialFigureStyle(fig,ax,legendColumns)
% Apply the manuscript figure layout without changing plotted data or colors.
set(fig,'Color','w','Units','centimeters','Position',[2 2 20 19], ...
    'PaperPositionMode','auto');
set(ax,'FontName','Arial','FontSize',10,'LineWidth',0.8, ...
    'TickDir','out','Layer','top');
set(get(ax,'XLabel'),'FontName','Arial','FontSize',11);
set(get(ax,'YLabel'),'FontName','Arial','FontSize',11);
set(get(ax,'Title'),'FontName','Arial','FontSize',12,'FontWeight','bold');
if legendColumns > 0
    lgd = legend(ax,'show');
    set(lgd,'Location','southoutside','Orientation','vertical', ...
        'NumColumns',2,'FontName','Arial','FontSize',8, ...
        'Box','off','Interpreter','tex');
end
end
