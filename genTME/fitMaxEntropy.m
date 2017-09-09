%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% <TME>
% Copyright (C) 2017 Gamaleldin F. Elsayed and John P. Cunningham 
%       (see full notice in README)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [maxEntropy] = fitMaxEntropy(params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function solves for the Lagrangian multipliers of the maximum
% entropy distribution.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
%     - params.
%       - margCov: specified set of marginal covariance across tensor modes.
%       - meanTensor: specified mean tensor.
% Outputs:
%       - maxEntropy:
%               .Lagrangians: are the eigenvalues of the largrangian
%                multipliers of the optimization program
%               .eigVectors: are the eigVectors of the largrangian
%                multipliers of the optimization program.
%               .objCost: is the final objective cost.
%               .logObjperIter: is the transformed log objective cost at each
%                iteration. Note, optimization is performed only on the
%                log objective because the original objective can be
%                problematic and slowly converges when the specified
%                marginal covariance matrices are low rank. The optimal
%                solution of the log objective and the original objective 
%                is the same and both the log objective and original objective
%                values at the global optimum should be 0.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [maxEntropy] = fitMaxEntropy(params)
margCov = params.margCov;
maxEntropy.meanTensor = params.meanTensor; % max entropy does not depend on the mean so just save it for th esampling func
tensorSize = length(margCov);                                             % tensor size; ie the number of different dimensions of the tensor
dim = nan(tensorSize, 1);                                                  % tensor dimensions
eigVectors = cell(tensorSize, 1);                                          % eigenVectors of each of the specified marginal covariances
eigValues = cell(tensorSize, 1);                                           % eigenValues of each of the specified marginal covariances
trSigma = nan(tensorSize, 1);                                              % sum of each of the eigenValues of each of the specified marginal covariances
dim = size(maxEntropy.meanTensor);
for i = 1:tensorSize
    if ~isempty(margCov{i})
       Tr = trace(margCov{i});
       break
    end
end

for i = 1:tensorSize                                                       % load all the inputs
    Sigma = margCov{i};
    if isempty(Sigma)
        Sigma = eye(dim(i))*(Tr/dim(i));
    end
    [Q, S] = svd(Sigma);
    [S, ix] = sort(diag(S), 'descend');
    Q = Q(:, ix);
    eigVectors{i} = Q;
    eigValues{i} = S;
    trSigma(i) = trace(Sigma);
end
maxEntropy.eigVectors = eigVectors;
%% the marginal covariances should all have the same trace (i.e. the sum of their eigenvalues should be equal)
if ~(sum((trSigma-mean(trSigma))>=-(sum(dim)*sqrt(eps)) & (trSigma-mean(trSigma))<=(sum(dim)*sqrt(eps)))==length(trSigma))
    error('the covariance matrices should have exactly the same trace')
end

%%
 % if the marginal covariances are low rank then the number of variables 
 % that we solve for are less. If full rank the number of variables that we 
 % solve for are equal to the sum of the tensor dimensions.

figFlg = false;                                                             % display summary figure flag
tensorSize = length(eigValues);                                            % tensor size; ie the number of different dimensions of the tensor
dim = nan(tensorSize,1);                                                   % tensor dimensions
tensorIxs = 1:tensorSize;                                                  
threshold = -10;                                                           % if an eigenvalue is below this threshold it is considered 0. 
for x = tensorIxs
   dim(x) = length(eigValues{x});
end
preScale = (sum(eigValues{1})./mean(dim));                                 % prescale the eigenvalues for numerical stability
logEigValues = cell(tensorSize,1);                                         % the log of the eigenvalues
optDim = nan(tensorSize,1);                                                % true number of variables that we solve for, which is equal to the sum of the ranks of the marginal covariances                                                
for x = tensorIxs
    logEigValues{x} = log(eigValues{x}./preScale);
    logEigValues{x} = logEigValues{x}(logEigValues{x}>threshold); % eigenvalues should be order apriori
    optDim(x) = length(logEigValues{x});
end

%% instead of solving for the largrangians directly we optimize latent variables that is equal to the log of the lagrangians
%% initialization of the optimization variables                                         
logL0 = cell(tensorSize,1);                                                
for x = tensorIxs
    nxSet = tensorIxs(tensorIxs~=x);
    logL0{x} = log(sum(optDim(nxSet)))-logEigValues{x};
end    

%% this is the optimization step
maxiter = 10000;                                                           % maximum allowed iterations
[logL, logObjperIter] = minimize(vertcat(logL0{:}) ,...
    'logObjectiveMaxEntropyTensor' , maxiter, logEigValues);               % this function performs all the optimzation heavy lifting
L = exp(logL);                                                             % calculates the optimal Largrangians from the latent by taking the exponential
Lagrangians = cell(tensorSize,1);                                          % save the lagrangians to the output 
for x = tensorIxs
    Lagrangians{x} = [L(sum(optDim(1:x-1))+(1:optDim(x)));...
        Inf*ones(dim(x)-optDim(x),1)]./preScale;                           % add the lagrangians known solution (Infinity) of the zero marginal covariance eigenvalues (if low rank)
end
%% save and display summary
objCost = objectiveMaxEntropyTensor(vertcat(Lagrangians{:}), eigValues);
logObjCost = logObjectiveMaxEntropyTensor(logL, logEigValues);
maxEntropy.Lagrangians = Lagrangians;
maxEntropy.logObjperIter = logObjperIter;
maxEntropy.objCost = objCost;
% fprintf('\n final cost value: \n')
% fprintf(' - gradient inconsistency with numerical gradient = %.5f \n',...
%     checkgrad('logObjectiveMaxEntropyTensor', randn(sum(optDim),1), 1e-5, logEigValues))
fprintf(' - final cost value = %1.5e \n', logObjCost)


% fprintf('\n cost: \n')
% fprintf(' - gradient inconsistency with numerical gradient = %.5f \n',...
%     checkgrad('objectiveMaxEntropyTensor', abs(randn(sum(dim),1)), 1e-5, eigValues));
% fprintf(' - cost at convergence = %.5f \n', objCost)

if objCost>1e-5
    warning('algorithm did not converege, results may be inaccuarate')
end


if figFlg
figure;
plot(logObjperIter, 'r.-');
xlabel('Iteration #');
ylabel('objective function value');
ylim([0 1])
set(gca, 'FontSize', 16)
end
end
