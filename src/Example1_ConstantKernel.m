%% PROGRAM: Legendre approximation-based stability test for distributed delay systems 
% CASE: Distributed delay with a constant kernel.
% Author: Alejandro Castaño Hernández
% Advisors: Dr. Mathieu Bajodek / Dr. Sabine Mondié

tic

%% Workspace initialization
clc
clear
hold on

% High-contrast, color-vision-conscious palette used consistently in all maps.
colorPositive = [0.000 0.447 0.698];        % blue (#0072B2)
colorNonPositive = [0.835 0.369 0.000];     % vermilion (#D55E00)
colorPending = [0.929 0.694 0.125];         % amber (#EDB120)
colorReview = [0.800 0.475 0.655];          % magenta/purple (#CC79A7)
colorDPartition = [0.25 0.25 0.25];         % dark gray for s=jw curve
colorDisagreement = [0.494 0.184 0.556];    % purple outline for diagnostic mismatch

%% Run configuration
% 'exploration' computes P_n with n=5 over a fine grid.
% 'criterion' computes a conservative prescribed-order bound and attempts P_n
% only when the order is below maxOrderCriterion. The norm bound may be
% conservative; points above the configured ceiling are logged, not classified.
analysisMode = 'exploration'; % set to 'criterion' for prescribed-order runs
nExploration = 5;
maxOrderCriterion = 80;
explorationGridIntervals = 140;
criterionGridIntervals = 12; % smaller grid because prescribed orders may be expensive
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
fprintf('Constant-kernel run mode: %s; grid: %d x %d values; output tag: %s\n', ...
    analysisMode,gridIntervals+1,gridIntervals+1,runTag);
progressSummaryFile = sprintf('ScanProgress_ConstantKernel_%s.csv',runTag);
progressMapFile = sprintf('StabilityMap_ConstantKernel_%s_PROGRESS.png',runTag);
runtimeErrorFile = sprintf('RuntimeError_ConstantKernel_%s.txt',runTag);

% All computations use double precision; tolerances are reported as numerical safeguards.
positiveDefinitePoints = zeros(0,2); % Points passing the finite-order positive-definiteness test.
excludedPoints = zeros(0,8); % [p1,p2,multAlg0,multGeom0,minAbsHlambda,rcondV,rcondX,rhoSym]
exclusionReason = cell(0,1);
pointsPassingPrecomputationFilters = 0;
pointsWithPnAssembled = 0; % P_n assembled before finiteness/symmetry diagnostics
evaluatedPoints = zeros(0,2);
nonPositiveDefinitePoints = zeros(0,2);
minEigenvalueSymPartEvaluated = zeros(0,1);
symmetryResidualEvaluated = zeros(0,1);
minAbsHlambdaEvaluated = zeros(0,1);
lowSpectralScaleEvaluated = false(0,1);
positiveDefiniteSymPartEvaluated = false(0,1);
statusEvaluated = cell(0,1);
numericalReviewPoints = zeros(0,9); % [p1,p2,nUsed,nSafe,minAbsHlambda,minEigPn,rhoSym,Ubound,pdRaw]
numericalReviewReason = cell(0,1);
orderUsedEvaluated = zeros(0,1);
safeOrderBoundEvaluated = zeros(0,1);
logSafeOrderBoundEvaluated = zeros(0,1);
UboundEvaluated = zeros(0,1);
orderPending = zeros(0,7); % [p1,p2,nSafe,logN,Ubound,mu,eta]
orderPendingReason = cell(0,1);
tolRecurrence = 0.5; % Empirical warning threshold; not a hard filter in either mode.
tolX = 1e-12;
tolSym = 1e-8;
tolZeroBase = 1e-8;
p1_min = -100; p1_max = 200;
p2_min = -100; p2_max = 200;
n = nExploration; % overwritten pointwise in criterion mode
nD = 3;  % Number of dynamics in the semi-analytical representation: U(tau), U(tau-h), and int_{0}^{h} U(tau+theta) dtheta.

h = 0.1;
p1_values = linspace(p1_min,p1_max,gridIntervals+1);
p2_values = linspace(p2_min,p2_max,gridIntervals+1);
for rowIndex = 1:numel(p1_values)
    p1 = p1_values(rowIndex);
    for p2 = p2_values
        try
        %% Distributed-delay system

        % Definition of the distributed-delay system

        % Example 1: Scalar case over [-200,200] x [-200,200], with h = 0.1
        A0 = p1;
        A1 = p2;
        G  = 1;

        % Definition of the required matrices
        nx = length(A0);
        In = eye(nx);
        I0 = zeros(nx);


        %% Lyapunov matrix computation
        % Matrix L:
        L = [ kron(A0',In)  kron(A1',In)  kron(G',In);
             -kron(In,A1') -kron(In,A0') -kron(In,G');
              kron(In,In)  -kron(In,In)   kron(I0,I0)];

        %% NUMERICAL FILTER: zero must be a simple eigenvalue
        % The algebraic and geometric multiplicities must both be equal to one.
        % This is a numerical check and depends on the selected tolerances.
        tolZero = tolZeroBase * max(1, norm(L,2));

        lambdaL = eig(L);
        multAlg0 = sum(abs(lambdaL) <= tolZero);

        singL = svd(L);
        multGeom0 = sum(singL <= tolZero);

        % If zero is not simple, exclude this point from the parameter sweep.
        if multAlg0 ~= 1 || multGeom0 ~= 1
            excludedPoints(end+1,:) = [p1, p2, multAlg0, multGeom0, NaN, NaN, NaN, NaN];
            exclusionReason{end+1,1} = 'Zero eigenvalue is not numerically simple';
            continue;
        end

        % Spectral diagonalization (this is not a general Jordan canonical form).
        % The filter rejects cases in which zero is not numerically simple.
        [V,D] = eig(L);

        % Reject numerically ill-conditioned spectral decompositions.
        rcP = rcond(V);
        eigenpairResidual = norm(L*V - V*D, 'fro') / max(1,norm(L,'fro'));
        if ~isfinite(rcP) || rcP < 1e-10 || ...
                ~isfinite(eigenpairResidual) || eigenpairResidual > 1e-8
            excludedPoints(end+1,:) = [p1, p2, multAlg0, multGeom0, NaN, rcP, NaN, NaN];
            exclusionReason{end+1,1} = 'Eigenvector conditioning or eigenpair residual failed';
            continue;
        end

        % Ensure that D contains exactly one mode treated as zero.
        idxZero = find(abs(diag(D)) <= tolZero);
        if numel(idxZero) ~= 1
            excludedPoints(end+1,:) = [p1, p2, multAlg0, multGeom0, NaN, rcP, NaN, NaN];
            exclusionReason{end+1,1} = 'Zero-mode index is not unique after eigendecomposition';
            continue;
        end

        lambdaD = diag(D);
        idxNonzero = abs(lambdaD) > tolZero;
        if any(idxNonzero)
            minAbsHlambda = min(abs(h*lambdaD(idxNonzero)));
        else
            minAbsHlambda = 0;
        end
        % A small nonzero |h*lambda| can cause cancellation in some recurrences.
        % The threshold 0.5 is an empirical warning from the n=5 constant-kernel
        % checks, not a mathematical admissibility condition. Do not discard such
        % points from the parameter map: evaluate P_n and report the warning separately.
        lowSpectralScaleWarning = ~isfinite(minAbsHlambda) || minAbsHlambda < tolRecurrence;
        if ~isfinite(minAbsHlambda)
            excludedPoints(end+1,:) = [p1, p2, multAlg0, multGeom0, minAbsHlambda, rcP, NaN, NaN];
            exclusionReason{end+1,1} = 'Nonzero spectral scale is nonfinite';
            continue;
        end

        % Matrix M:
        M  = [ kron(In,In)   kron(I0,I0)   kron(I0,I0); 
               kron(I0,I0)   kron(I0,I0)   kron(In,In);
               kron(A0',In)  kron(A1',In)  kron(G',In)];

        % Matrix N:
        N  = [ kron(I0,I0)  -kron(In,In)   kron(I0,I0);
               kron(I0,I0)   kron(I0,I0)   kron(I0,I0);
               kron(In,A1')  kron(In,A0')  kron(In,G')];

        % Matrix E
        E  = [kron(I0,I0) kron(In,In) kron(I0,I0)];

        % Matrix X
        X  = M + [-E' N]*expm([kron(I0,I0) E; zeros(nD*nx^2,nx^2) L]*h)*[zeros(nx^2,nD*nx^2); eye(nD*nx^2)];
        % Check the boundary matrix before solving its linear system.
        rcX = rcond(X);
        if ~isfinite(rcX) || rcX < tolX
            excludedPoints(end+1,:) = [p1, p2, multAlg0, multGeom0, minAbsHlambda, rcP, rcX, NaN];
            exclusionReason{end+1,1} = 'Boundary matrix X is singular or too ill-conditioned';
            continue;
        end
        pointsPassingPrecomputationFilters = pointsPassingPrecomputationFilters + 1;

        % Boundary conditions: solve X*x0=b instead of forming inv(X).
        W  = eye(nx);
        x0 = X\[I0(:); I0(:); -W(:)];

        % Computation of U0
        tau = 0;
        U   = [kron(In,In) kron(I0,I0) kron(I0,I0)]*expm(tau*L)*x0;
        U0  = double(reshape(U, nx, nx));

        if strcmpi(analysisMode,'criterion')
            % Conservative bound for ||U||_h. The semi-analytical solution gives
            % U(tau)=E1*exp(tau*L)*x0 for tau in [0,h]; symmetry covers [-h,0].
            muSystem = norm(A0,2) + norm(A1,2) + h*norm(G,2);
            etaOrder = h*norm(A1,2) + h^2*norm(G,2);
            muLogL = max(real(eig((L+L')/2)));
            logUbound = log(max(norm(x0,2),realmin)) + max(0,h*muLogL);
            if isfinite(logUbound) && logUbound < log(realmax)-2
                Ubound = exp(logUbound);
            else
                Ubound = Inf;
            end
            [nSafe,orderInfo] = compute_legendre_order_bound(muSystem,etaOrder,Ubound,h,eye(nx));
            if ~isfinite(nSafe) || max(2,nSafe) > maxOrderCriterion
                orderPending(end+1,:) = [p1,p2,nSafe,orderInfo.logOrderBound,Ubound,muSystem,etaOrder];
                orderPendingReason{end+1,1} = sprintf('Conservative prescribed order unavailable or exceeds configured limit %d (%s)',maxOrderCriterion,orderInfo.status);
                continue;
            end
            n = max(2,nSafe); % the implemented recurrence initialization requires n>=2
        else
            % Do not calculate the order bound for every point of the n=5 scan:
            % it is expensive and is not needed for the exploratory map.
            n = nExploration;
            nSafe = NaN;
            Ubound = NaN;
            orderInfo = struct('status','not-computed-in-exploration','logOrderBound',NaN);
        end

        %% Generalized inverse diagonal for the recurrence relations
        % For lambda ~= 0: Di(i,i) = 1/lambda.
        % For the only lambda = 0 mode, set Di(i,i) = 0; the limits of the
        % recurrence functions are incorporated explicitly below.
        lambdaD = diag(D);
        dInv = zeros(size(lambdaD));
        idxNoZero = abs(lambdaD) > tolZero;
        dInv(idxNoZero) = 1 ./ lambdaD(idxNoZero);
        Di = diag(dInv);

        % Eigenvector-basis inverse applied by linear solve, not by explicit inv(V).
        Vinv = V \ eye(size(V));

        %% Recurrence relations for Gamma1 and Lambda1
        Gamma    = cell(n,1);
        Gamma{1} = Di*( expm(h*D) - eye(nD*nx^2) );                    %% Gamma[0]
        Gamma{2} = Di*( expm(h*D) + eye(nD*nx^2) ) - 2/h*Di*Gamma{1};  %% Gamma[1]

        Lambda    = cell(n,1);
        Lambda{1} = Di*( Gamma{1} - h*eye(nD*nx^2) );                  %% Lambda[0]
        Lambda{2} = Di*( Gamma{2} );                                   %% Lambda[1]

        % Compute the nonzero-mode recurrences first. The zero-mode limits must
        % be imposed AFTER recurrence evaluation: inserting Gamma_0(0)=h before
        % this loop would spuriously propagate nonzero values to Gamma_2,Gamma_4,...
        for k=2:n-1
            Gamma{k+1}  = Gamma{k-1} - 2*(2*k-1)/h*Di*Gamma{k};
            Lambda{k+1} = Di*Gamma{k+1};
        end

        % Exact zero-mode values obtained directly from the defining integrals.
        for iz = idxZero(:)'
            for kk = 1:n
                Gamma{kk}(iz,iz) = 0;
                Lambda{kk}(iz,iz) = 0;
            end
            Gamma{1}(iz,iz) = h;              % Gamma_0(0)
            Lambda{1}(iz,iz) = h^2/2;         % Lambda_0(0)
            if n >= 2
                Lambda{2}(iz,iz) = h^2/6;      % Lambda_1(0)
            end
        end

        Qn = zeros(nx,n*nx);
        Rn = zeros(nx,n*nx);

        for k=0:n-1
            Qk = reshape([kron(In,In) kron(I0,I0) kron(I0,I0)]*V*Gamma{k+1}*Vinv*x0, nx, nx);
            Qn(:,k*nx+1:(k+1)*nx) = Qk'*A1;

            Rk = reshape([kron(In,In) kron(I0,I0) kron(I0,I0)]*V*Lambda{k+1}*Vinv*x0, nx, nx);
            Rn(:,k*nx+1:(k+1)*nx) = Rk'*G;
        end

        %% Recurrence relation for GammaBar
        GammaBar = cell(n);
        GammaBar{1,1} =  Di*( Gamma{1} - h*eye(nD*nx^2) );
        GammaBar{1,2} = -Di*Gamma{2};
        GammaBar{2,2} =  Di*( ( 2/h*Di - eye(nD*nx^2) )*Gamma{2} - h/3*eye(nD*nx^2) );

        for j=1:n
            for k=1:n
                if k >= max(3,j)
                    GammaBar{j,k} = GammaBar{j,k-2} + 2*(2*k-3)/h*Di*GammaBar{j,k-1};
                    % Add the Kronecker-delta terms
                    if k == j
                        GammaBar{j,k} = GammaBar{j,k} - h/(2*j-1)*Di;
                    end
                    % Add the Kronecker-delta terms
                    if k == j+2
                        GammaBar{j,k} = GammaBar{j,k} + h/(2*j-1)*Di;
                    end
                else
                    if k < j
                        GammaBar{j,k} = (-1)^(j+k)*GammaBar{k,j};
                    end
                end
            end
        end

        % Exact zero-mode overwrite for EVERY coefficient of GammaBar.
        % Correcting only GammaBar{1,1} misses, for example,
        % GammaBar_{0,1}(0)=-h^2/6 and GammaBar_{1,0}(0)=h^2/6.
        for iz = idxZero(:)'
            for j0 = 0:n-1
                for k0 = 0:n-1
                    if k0 == 0
                        val0 = h^2/2*double(j0==0) + h^2/6*double(j0==1);
                    else
                        val0 = h^2/(2*(2*k0+1)) * ...
                            (double(j0==k0+1)-double(j0==k0-1))/(2*j0+1);
                    end
                    GammaBar{j0+1,k0+1}(iz,iz) = val0;
                end
            end
        end

        %% Recurrence relations for Zeta and ZetaBar

        % Recurrence relation for Zeta:
        Zeta = cell(n);
        for j=1:n
            for k=1:n
                if k==1
                   Zeta{j,k} = h*Di*Gamma{j};

                   if j==1
                       Zeta{j,k} = Zeta{j,k} - h^2*Di*eye(nD*nx^2);
                   end

                else
                   Zeta{j,k} = zeros(nD*nx^2);
                end

            end
        end 

        % Recurrence relation for ZetaBar
        ZetaBar = cell(n);
        for j=1:n
            for k=1:n
                ZetaBar{j,k} = ((-1)^(j-1))*Di*Gamma{k}*Gamma{j}*expm(-h*D);
                if j==1 && k==1
                    ZetaBar{j,k} = ZetaBar{j,k} - h^2*Di*eye(nD*nx^2);
                end
            end
        end

        % Exact zero-mode overwrite for the mixed integral families.
        for iz = idxZero(:)'
            for j0 = 1:n
                for k0 = 1:n
                    Zeta{j0,k0}(iz,iz) = 0;
                    ZetaBar{j0,k0}(iz,iz) = 0;
                end
            end
            Zeta{1,1}(iz,iz) = h^3/2;
            if n >= 2
                Zeta{2,1}(iz,iz) = h^3/6;
                ZetaBar{1,2}(iz,iz) = h^3/6;
                ZetaBar{2,1}(iz,iz) = -h^3/6;
            end
        end


        %% Recurrence relation for D
        DD = cell(n);
        for j=1:n
            for k=1:n
                if k==1
                    DD{j,k} = h*Di^2*Gamma{j};
                    
                    if j==1
                        DD{j,k} = -h^3/2*Di - h^2*Di^2 + h*Di^2*Gamma{j};
                    end

                    if j==2
                        DD{j,k} = -h^3/6*Di^2+ h*Di^2*Gamma{j};
                    end

                else
                    DD{j,k} = zeros(nD*nx^2);
                end


            end
        end

        %% Recurrence relation for DBar
        DBar = cell(n);

        for j=1:n
            for k=1:n
                DBar{j,k} = -(-1)^(j-1)*Di^2*Gamma{k}*Gamma{j}*expm(-h*D);

                if j==1
                   DBar{j,k} = DBar{j,k} + Di^2*h*Gamma{k};
                end

                if j==1 && k==1
                   DBar{j,k} = DBar{j,k} - h^3/2*Di;
                end

                if j==1 && k==2
                   DBar{j,k} = DBar{j,k} - h^3/6*Di;
                end
                
            end
        end

        % Exact zero-mode overwrite for the quadratic integral families.
        for iz = idxZero(:)'
            for j0 = 1:n
                for k0 = 1:n
                    DD{j0,k0}(iz,iz) = 0;
                    DBar{j0,k0}(iz,iz) = 0;
                end
            end
            DD{1,1}(iz,iz) = h^4/6;
            DBar{1,1}(iz,iz) = h^4/12;
            if n >= 2
                DD{2,1}(iz,iz) = h^4/12;
                DBar{1,2}(iz,iz) = h^4/12;
                DBar{2,2}(iz,iz) = h^4/36;
            end
            if n >= 3
                DD{3,1}(iz,iz) = h^4/60;
                DBar{3,1}(iz,iz) = -h^4/60;
            end
        end


                %% Pn matrices
        Sn = zeros(n*nx);
        TnSym = zeros(n*nx);
        Dn = zeros(n*nx);
        for j = 0:n-1
            for k = 0:n-1
                Tjk = reshape([kron(In,In) kron(I0,I0) kron(I0,I0)]*V*GammaBar{j+1,k+1}*Vinv*x0, nx, nx);
                TnSym(j*nx+1:(j+1)*nx,k*nx+1:(k+1)*nx) = A1'*(Tjk+(-1)^(j+k)*Tjk')*A1;

                Sjk    = reshape([kron(In,In) kron(I0,I0) kron(I0,I0)]*V*Zeta{j+1,k+1}*Vinv*x0, nx, nx);
                SjkBar = reshape([kron(In,In) kron(I0,I0) kron(I0,I0)]*V*ZetaBar{j+1,k+1}*Vinv*x0, nx, nx);
                Sn(j*nx+1:(j+1)*nx,k*nx+1:(k+1)*nx) = A1'*(Sjk+SjkBar')*G;

                Djk    = reshape([kron(In,In) kron(I0,I0) kron(I0,I0)]*V*DD{j+1,k+1}*Vinv*x0, nx, nx);
                DjkBar = reshape([kron(In,In) kron(I0,I0) kron(I0,I0)]*V*DBar{j+1,k+1}*Vinv*x0, nx, nx);
                Dn(j*nx+1:(j+1)*nx,k*nx+1:(k+1)*nx) = G'*(Djk+DjkBar')*G;

            end
        end


        %% IIn term
        IIn = 1/h*kron(diag(2*(0:n-1)+1),eye(nx));

        Pn = [U0 Qn+Rn; Qn'+Rn' TnSym+Sn+Sn'+0.5*(Dn+Dn')+(IIn\eye(size(IIn)))];
        pointsWithPnAssembled = pointsWithPnAssembled + 1;
        %Pn = U0;

        %% Finite-order P_n test and symmetry diagnostic
        if any(~isfinite(Pn(:)))
            excludedPoints(end+1,:) = [p1, p2, multAlg0, multGeom0, minAbsHlambda, rcP, rcX, NaN];
            exclusionReason{end+1,1} = 'P_n contains nonfinite entries';
            continue;
        end
        rhoSym = norm(Pn-Pn','fro')/max(1,norm(Pn,'fro'));
        if ~isfinite(rhoSym)
            excludedPoints(end+1,:) = [p1, p2, multAlg0, multGeom0, minAbsHlambda, rcP, rcX, rhoSym];
            exclusionReason{end+1,1} = 'P_n symmetry diagnostic is nonfinite';
            continue;
        end
        % The symmetric part is used for diagnostics, but a large residual is
        % NOT silently accepted as a valid P_n: such points are marked for review.
        Pn_sym = double((Pn + Pn')/2);
        [~,Px] = chol(Pn_sym);
        minEigPn = min(real(eig(Pn_sym)));
        evaluatedPoints(end+1,:) = [p1,p2];
        minEigenvalueSymPartEvaluated(end+1,1) = minEigPn;
        symmetryResidualEvaluated(end+1,1) = rhoSym;
        minAbsHlambdaEvaluated(end+1,1) = minAbsHlambda;
        lowSpectralScaleEvaluated(end+1,1) = lowSpectralScaleWarning;
        positiveDefiniteSymPartEvaluated(end+1,1) = (Px == 0);
        orderUsedEvaluated(end+1,1) = n;
        safeOrderBoundEvaluated(end+1,1) = nSafe;
        logSafeOrderBoundEvaluated(end+1,1) = orderInfo.logOrderBound;
        UboundEvaluated(end+1,1) = Ubound;

        symmetryWarning = rhoSym > tolSym;
        reviewFlag = symmetryWarning || lowSpectralScaleWarning;
        if reviewFlag
            numericalReviewPoints(end+1,:) = [p1,p2,n,nSafe,minAbsHlambda,minEigPn,rhoSym,Ubound,double(Px==0)];
            if symmetryWarning && lowSpectralScaleWarning
                numericalReviewReason{end+1,1} = 'P_n symmetry residual and low spectral scale both require review';
            elseif symmetryWarning
                numericalReviewReason{end+1,1} = 'P_n symmetry residual exceeds tolerance; the symmetric-part test is diagnostic only';
            else
                numericalReviewReason{end+1,1} = 'Low spectral scale: finite-order result is shown as exploratory and requires review';
            end
        end

        if strcmpi(analysisMode,'exploration')
            % Preserve the n=5 exploratory map. Classification here refers to
            % the symmetric-part Cholesky diagnostic; flagged points receive an
            % additional magenta marker and are not presented as certified points.
            if Px == 0
                positiveDefinitePoints(end+1,:) = [p1,p2];
                if reviewFlag
                    statusEvaluated{end+1,1} = 'positive-definite-symmetric-part-at-n5-review-required';
                else
                    statusEvaluated{end+1,1} = 'positive-definite-symmetric-part-at-n5';
                end
            else
                nonPositiveDefinitePoints(end+1,:) = [p1,p2];
                if reviewFlag
                    statusEvaluated{end+1,1} = 'not-positive-definite-symmetric-part-at-n5-review-required';
                else
                    statusEvaluated{end+1,1} = 'not-positive-definite-symmetric-part-at-n5';
                end
            end
        elseif reviewFlag
            % At criterion order, flagged points remain unclassified.
            if symmetryWarning && lowSpectralScaleWarning
                statusEvaluated{end+1,1} = 'criterion-order-pending-symmetry-and-spectral-review';
            elseif symmetryWarning
                statusEvaluated{end+1,1} = 'criterion-order-pending-symmetry-review';
            else
                statusEvaluated{end+1,1} = 'criterion-order-pending-spectral-review';
            end
        elseif Px == 0
            positiveDefinitePoints(end+1,:) = [p1,p2];
            statusEvaluated{end+1,1} = 'positive-definite-at-prescribed-order';
        else
            nonPositiveDefinitePoints(end+1,:) = [p1,p2];
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

    % Progress is visible while the scan runs. A checkpoint map/CSV is overwritten
    % periodically so a stop or later export problem does not leave the user with
    % no indication of the completed portion of the grid.
    if rowIndex==1 || mod(rowIndex,10)==0 || rowIndex==numel(p1_values)
        fprintf('Progress constant kernel: row %d/%d; P_n tests=%d; PD=%d; non-PD=%d; exclusions=%d\n', ...
            rowIndex,numel(p1_values),size(evaluatedPoints,1),size(positiveDefinitePoints,1),size(nonPositiveDefinitePoints,1),size(excludedPoints,1));
        progressTable = table(rowIndex,numel(p1_values),size(evaluatedPoints,1),size(positiveDefinitePoints,1), ...
            size(nonPositiveDefinitePoints,1),size(excludedPoints,1), ...
            'VariableNames',{'completedRows','totalRows','pointsWithPnDiagnostic','positiveDefiniteSymPart', ...
            'notPositiveDefiniteSymPart','excludedPoints'});
        writetable(progressTable,progressSummaryFile);
        fProgress = figure('Visible','off','Color','w');
        axProgress = axes(fProgress); hold(axProgress,'on');
        if ~isempty(positiveDefinitePoints), plot(axProgress,positiveDefinitePoints(:,1),positiveDefinitePoints(:,2),'.','Color',colorPositive,'MarkerSize',7); end
        if ~isempty(nonPositiveDefinitePoints), plot(axProgress,nonPositiveDefinitePoints(:,1),nonPositiveDefinitePoints(:,2),'.','Color',colorNonPositive,'MarkerSize',6); end
        if ~isempty(excludedPoints), plot(axProgress,excludedPoints(:,1),excludedPoints(:,2),'kx','MarkerSize',3); end
        if ~isempty(numericalReviewPoints), plot(axProgress,numericalReviewPoints(:,1),numericalReviewPoints(:,2),'d','Color',colorReview,'MarkerFaceColor','none','MarkerSize',4); end
        axis(axProgress,[p1_min p1_max p2_min p2_max]); axis(axProgress,'square'); grid(axProgress,'on'); box(axProgress,'on');
        xlabel(axProgress,'p_1'); ylabel(axProgress,'p_2');
        title(axProgress,sprintf('Constant kernel: partial n=%d scan, rows %d/%d',nExploration,rowIndex,numel(p1_values)),'Interpreter','none');
        print(fProgress,progressMapFile,'-dpng','-r150'); close(fProgress);
    end
end
%% Summary and audit trail
fprintf('\n===== CONSTANT-KERNEL FINITE-ORDER TEST =====\n');
totalGridPoints = numel(p1_values)*numel(p2_values);
fprintf('Total grid points: %d\n', totalGridPoints);
fprintf('Points passing precomputation filters: %d\n', pointsPassingPrecomputationFilters);
fprintf('Points reaching the P_n test: %d\n', size(evaluatedPoints,1));
if strcmpi(analysisMode,'criterion')
    fprintf('Analysis mode: prescribed-order criterion (conservative order bound).\n');
    fprintf('Positive definite at the tested conservative order: %d\n', size(positiveDefinitePoints,1));
    fprintf('Not positive definite at the tested conservative order: %d\n', size(nonPositiveDefinitePoints,1));
else
    fprintf('Analysis mode: exploratory finite-order screen, n=%d.\n', nExploration);
    fprintf('Positive definite at exploration order: %d\n', size(positiveDefinitePoints,1));
    fprintf('Not positive definite at exploration order: %d\n', size(nonPositiveDefinitePoints,1));
end
fprintf('Numerical exclusions before/at P_n: %d\n', size(excludedPoints,1));
fprintf('Points pending because the prescribed order was unavailable/over the ceiling: %d\n', size(orderPending,1));
fprintf('Points flagged for symmetry and/or spectral-scale review: %d\n', size(numericalReviewPoints,1));
fprintf('Empirical spectral-scale warning threshold: min |h*lambda_nonzero| < %.3g; warning only, not a hard exclusion.\n', tolRecurrence);
excludedFile = sprintf('ExcludedPoints_ConstantKernel_%s.csv',runTag);
if ~isempty(excludedPoints)
    excludedTable = array2table(excludedPoints, ...
        'VariableNames', {'p1','p2','multAlg0','multGeom0','minAbsHlambda','rcondV','rcondX','rhoSym'});
    excludedTable.Reason = exclusionReason;
else
    excludedTable = table('Size',[0 9], ...
        'VariableTypes',{'double','double','double','double','double','double','double','double','string'}, ...
        'VariableNames',{'p1','p2','multAlg0','multGeom0','minAbsHlambda','rcondV','rcondX','rhoSym','Reason'});
end
writetable(excludedTable,excludedFile);
evaluatedFile = sprintf('EvaluatedPoints_ConstantKernel_%s.csv',runTag);
if ~isempty(evaluatedPoints)
    nEvaluated = size(evaluatedPoints,1);
    assert(numel(orderUsedEvaluated)==nEvaluated && ...
        numel(safeOrderBoundEvaluated)==nEvaluated && ...
        numel(logSafeOrderBoundEvaluated)==nEvaluated && ...
        numel(UboundEvaluated)==nEvaluated && ...
        numel(minAbsHlambdaEvaluated)==nEvaluated && ...
        numel(lowSpectralScaleEvaluated)==nEvaluated && ...
        numel(minEigenvalueSymPartEvaluated)==nEvaluated && ...
        numel(symmetryResidualEvaluated)==nEvaluated && ...
        numel(positiveDefiniteSymPartEvaluated)==nEvaluated && ...
        numel(statusEvaluated)==nEvaluated, ...
        'ConstantKernel:EvaluatedColumnLengthMismatch', ...
        'Evaluated-point diagnostic columns do not have consistent lengths.');
    evaluatedTable = table(repmat({analysisMode},nEvaluated,1), ...
        evaluatedPoints(:,1), evaluatedPoints(:,2), orderUsedEvaluated(:), ...
        safeOrderBoundEvaluated(:), logSafeOrderBoundEvaluated(:), UboundEvaluated(:), ...
        minAbsHlambdaEvaluated(:), lowSpectralScaleEvaluated(:), minEigenvalueSymPartEvaluated(:), ...
        symmetryResidualEvaluated(:), positiveDefiniteSymPartEvaluated(:), statusEvaluated(:), ...
        'VariableNames', {'analysisMode','p1','p2','orderUsed','conservativeOrderBound', ...
        'logOrderBound','U_norm_upper_bound','minAbsHlambda','lowSpectralScaleWarning', ...
        'minEigPnSym','rhoSym','positiveDefiniteOfSymmetricPart','status'});
else
    evaluatedTable = table('Size',[0 13], ...
        'VariableTypes',{'string','double','double','double','double','double','double','double','logical','double','double','logical','string'}, ...
        'VariableNames',{'analysisMode','p1','p2','orderUsed','conservativeOrderBound', ...
        'logOrderBound','U_norm_upper_bound','minAbsHlambda','lowSpectralScaleWarning', ...
        'minEigPnSym','rhoSym','positiveDefiniteOfSymmetricPart','status'});
end
writetable(evaluatedTable,evaluatedFile);
reviewFile = sprintf('NumericalReviewPoints_ConstantKernel_%s.csv',runTag);
if ~isempty(numericalReviewPoints)
    numericalReviewTable = array2table(numericalReviewPoints, 'VariableNames', ...
        {'p1','p2','orderUsed','conservativeOrderBound','minAbsHlambda','minEigPnSym','rhoSym','U_norm_upper_bound','positiveDefiniteOfSymmetricPart'});
    numericalReviewTable.Reason = numericalReviewReason;
else
    numericalReviewTable = table('Size',[0 10], ...
        'VariableTypes',{'double','double','double','double','double','double','double','double','double','string'}, ...
        'VariableNames',{'p1','p2','orderUsed','conservativeOrderBound','minAbsHlambda','minEigPnSym','rhoSym','U_norm_upper_bound','positiveDefiniteOfSymmetricPart','Reason'});
end
writetable(numericalReviewTable,reviewFile);

pendingFile = sprintf('PendingCriterionOrder_ConstantKernel_%s.csv',runTag);
if ~isempty(orderPending)
    pendingOrderTable = array2table(orderPending, 'VariableNames', ...
        {'p1','p2','conservativeOrderBound','logOrderBound','U_norm_upper_bound','muSystem','eta'});
    pendingOrderTable.Reason = orderPendingReason;
else
    pendingOrderTable = table('Size',[0 8], ...
        'VariableTypes',{'double','double','double','double','double','double','double','string'}, ...
        'VariableNames',{'p1','p2','conservativeOrderBound','logOrderBound','U_norm_upper_bound','muSystem','eta','Reason'});
end
writetable(pendingOrderTable,pendingFile);

% Save a compact summary alongside the point-wise audit tables.
summaryFile = sprintf('Summary_ConstantKernel_%s.csv',runTag);
summaryTable = table({analysisMode}, gridIntervals+1, gridIntervals+1, totalGridPoints, ...
    pointsPassingPrecomputationFilters, pointsWithPnAssembled, size(evaluatedPoints,1), ...
    size(positiveDefinitePoints,1), size(nonPositiveDefinitePoints,1), size(excludedPoints,1), ...
    size(numericalReviewPoints,1), size(orderPending,1), ...
    'VariableNames', {'analysisMode','gridP1Count','gridP2Count','totalGridPoints', ...
    'pointsPassingPrecomputationFilters','pointsWithPnAssembled','pointsWithPnDiagnostic', ...
    'positiveDefiniteSymmetricPartDiagnostic','notPositiveDefiniteSymmetricPartDiagnostic', ...
    'numericallyExcluded','flaggedForNumericalReview','pendingOrder'});
writetable(summaryTable,summaryFile);

% Stability map: separate P_n classifications from any additional D-partition overlay.
figMap = figure('Color','w','Units','centimeters','Position',[2 2 20 19], ...
    'PaperPositionMode','auto');
axMap = axes(figMap); hold(axMap,'on');
if strcmpi(analysisMode,'criterion')
    plotLabel = 'P_n positive definite at prescribed order';
    plotTitle = 'Constant kernel: prescribed-order criterion scan';
else
    plotLabel = sprintf('Sym(P_%d) positive definite (exploratory diagnostic)',nExploration);
    plotTitle = sprintf('Constant kernel: symmetric-part diagnostic (n=%d)',nExploration);
end
if ~isempty(positiveDefinitePoints)
    plot(positiveDefinitePoints(:,1),positiveDefinitePoints(:,2),'.','Color',colorPositive, ...
        'MarkerSize',8,'DisplayName',plotLabel);
end
if ~isempty(nonPositiveDefinitePoints)
    plot(nonPositiveDefinitePoints(:,1),nonPositiveDefinitePoints(:,2),'.','Color',colorNonPositive, ...
        'MarkerSize',7,'DisplayName','Sym(P_n) not positive definite at tested order');
end
if ~isempty(excludedPoints)
    plot(excludedPoints(:,1),excludedPoints(:,2),'kx', ...
        'MarkerSize',4,'DisplayName','Excluded by numerical/admissibility filters');
end
if ~isempty(orderPending)
    plot(orderPending(:,1),orderPending(:,2),'s','Color',colorPending, ...
        'MarkerSize',4,'DisplayName','Prescribed order pending');
end
if ~isempty(numericalReviewPoints)
    plot(numericalReviewPoints(:,1),numericalReviewPoints(:,2),'d', ...
        'Color',colorReview,'MarkerFaceColor','none','MarkerSize',5, ...
        'DisplayName','Computed; numerical review required');
end
% Restore the original D-partition boundaries for the scalar constant-kernel model.
% These curves belong to the same (p_1,p_2) parameterization as this example.
p1_s0 = linspace(p1_min,p1_max,200);
p2_s0 = -p1_s0 + G*h;
plot(p1_s0,p2_s0,'k--','LineWidth',1.0,'DisplayName','D-partition: s=0');
dPartitionJwPoints = zeros(0,2);
for omega = 0.01:0.01:70*pi
    den = omega*sin(omega*h);
    if abs(den) < 1e-12
        continue;
    end
    p2_jw = (-omega^2+G-G*cos(omega*h))/den;
    p1_jw = (-p2_jw*omega*cos(omega*h)+G*sin(omega*h))/omega;
    if isfinite(p1_jw) && isfinite(p2_jw)
        dPartitionJwPoints(end+1,:) = [p1_jw,p2_jw]; %#ok<AGROW>
    end
end
if ~isempty(dPartitionJwPoints)
    plot(dPartitionJwPoints(:,1),dPartitionJwPoints(:,2),'.','Color',colorDPartition,'MarkerSize',4, ...
        'DisplayName','D-partition: s=j\omega');
end
axis(axMap,[p1_min p1_max p2_min p2_max]); axis(axMap,'square');
grid(axMap,'on'); box(axMap,'on');
xlabel(axMap,'p_1'); ylabel(axMap,'p_2');
title(axMap,plotTitle,'Interpreter','none');
applyConstantKernelFigureStyle(figMap,axMap,2);
mapFile = sprintf('StabilityMap_ConstantKernel_%s.png',runTag);
print(figMap,mapFile,'-dpng','-r300');
fprintf('Saved parameter map: %s\n',mapFile);
if exist(progressMapFile,'file'), delete(progressMapFile); end
if exist(progressSummaryFile,'file'), delete(progressSummaryFile); end

toc


function applyConstantKernelFigureStyle(fig,ax,legendColumns)
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
