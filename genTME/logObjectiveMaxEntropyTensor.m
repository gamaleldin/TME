%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% <TME>
% Copyright (C) 2017 Gamaleldin F. Elsayed and John P. Cunningham 
%       (see full notice in README)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [f, gradf_logL] = logObjectiveMaxEntropyTensor(logL, logEigValues)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function evaluates the log transformed objective function of the
% maximum entropy covariance eigenvalues. This transformed cost function is 
% better in cases when data is low rank. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
%       - logL: is the vector of the stacked log transformed 
%         eigenvalues of the lagrangian matrices
%       - logEigValues: cell with each element containing the vector of
%         log transformed eigenvalues of the specified marginal covariance 
%         matrices.
% Outputs:
%       - f: is the log transformed objective cost function evaluated at 
%         the input vector logL.
%       - gradf_logL: is the gradient of the log transformed objective cost
%         function evaluated at the input vector logL. This gradient is
%         taken with respect of the log tramsformed latent of the
%         lagrangian matrices eigenvalues (logL).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f, gradf_logL] = logObjectiveMaxEntropyTensor(logL, logEigValues)
   normalizeTerm = norm((vertcat(logEigValues{:}))).^2;                    % normalization value for the objective function and the gradient
   tensorSize = length(logEigValues);                                      % tensor size; i.e. the number of different dimensions of the tensor
   dim = nan(1,tensorSize);                                                % tensor dimensions
   Lagrangians = cell(tensorSize, 1);                                
   logLs = cell(tensorSize, 1);
   tensorIxs = 1:tensorSize;                                    
   for x = tensorIxs
       dim(x) = length(logEigValues{x});
       logLs{x} = logL(sum(dim(1:x-1))+(1:dim(x)));
       Lagrangians{x} = exp(logLs{x});
   end
   %% the building blocks of the gradient
   LsTensor = diagKronSum(Lagrangians);
   LsTensor = reshape(LsTensor, dim(:).');                                 % kronecker sum of the lagrangian matrices eigenvalues
   invLsTensor = 1./LsTensor;                                              % elementwise inverse of the above
   invSquareLsTensor = 1./LsTensor.^2;                                     % elementwise inverse square of the above
   Er = cell(tensorSize, 1);
   logSums = cell(tensorSize, 1);
   fx = cell(tensorSize, 1);                                               % the log transformed cost decomposed to different tensor dimensions
   for x = tensorIxs
       nxSet = tensorIxs(tensorIxs ~= x);
       logSums{x} = log(sumTensor(invLsTensor, nxSet));                    % elementwise log of invLsTensor
       Er{x} = reshape(logEigValues{x}, size(logSums{x}))- logSums{x};     % error with respect to each marginal covariance eigenvalue
       fx{x} = reshape(Er{x}, dim(x), 1).^2;                               
   end
   f = sum(vertcat(fx{:}))./normalizeTerm;                                 % the log transformed objective value
   %% build the gradient from the blocks
   gradf_logL = nan(sum(dim),1);                                           % gradient
   for x = tensorIxs
       nxSet = tensorIxs(tensorIxs ~= x);                                  
       gradfx_logLx = reshape(bsxfun(@times, ...
           2.*Er{x}./sumTensor(invLsTensor, nxSet),...
           sumTensor(invSquareLsTensor, nxSet)),...
           dim(x),1).*Lagrangians{x};
       
       gradfy_logLx = nan(dim(x), tensorSize-1);
       z = 1;
       for y = nxSet(:).'
           nySet = tensorIxs(tensorIxs ~= y);
           nxySet = tensorIxs((tensorIxs ~= x) & (tensorIxs ~= y));
           
           gradfy_logLx(:, z) = reshape(sumTensor(bsxfun(@times, ...
                2.*Er{y}./sumTensor(invLsTensor, nySet),...
                sumTensor(invSquareLsTensor, nxySet)), y),...
                dim(x),1).*Lagrangians{x};           
           z = z+1;
       end   
       gradf_logLx = sum([gradfx_logLx gradfy_logLx], 2);
       gradf_logL(sum(dim(1:x-1))+(1:dim(x))) = gradf_logLx;
   end
   gradf_logL = gradf_logL./normalizeTerm;                                 
end