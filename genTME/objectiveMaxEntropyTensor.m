%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% <TME>
% Copyright (C) 2017 Gamaleldin F. Elsayed and John P. Cunningham 
%       (see full notice in README)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [f, gradf_L] = objectiveMaxEntropyTensor(L, eigValues)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function evaluates the objective function of the maximum entropy 
% covariance eigenvalues problem
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
%       - L: is the vector of the stacked eigenvalues of the lagrangian
%         matrices.
%       - eigValues: cell with each element containing the vector of
%         eigenvalues of the specified marginal covariance matrices.
%         
% Outputs:
%       - f: is the objective cost function evaluated at the input vector
%         L.
%       - gradf_L: is the gradient of the objective cost function evaluated
%         at the input vector L with respect to L. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f, gradf_L] = objectiveMaxEntropyTensor(L, eigValues)
   normalizeTerm = norm((vertcat(eigValues{:}))).^2;                       % normalization value for the objective function and the gradient
   tensorSize = length(eigValues);                                         % tensor size; i.e. the number of different dimensions of the tensor
   dim = nan(1,tensorSize);                                                % tensor dimensions
   Lagrangians = cell(tensorSize, 1);                                
   tensorIxs = 1:tensorSize;                                    
   for x = tensorIxs
       dim(x) = length(eigValues{x});
       Lagrangians{x} = L(sum(dim(1:x-1))+(1:dim(x)));
   end
   %% the building blocks of the gradient
   LsTensor = diagKronSum(Lagrangians);
   LsTensor = reshape(LsTensor, dim(:).');                                 % kronecker sum of the lagrangian matrices eigenvalues
   invLsTensor = 1./LsTensor;                                              % elementwise inverse of the above
   invSquareLsTensor = 1./LsTensor.^2;                                     % elementwise inverse square of the above
   Er = cell(tensorSize, 1);
   fx = cell(tensorSize, 1);                                               % the cost decomposed to different tensor dimensions
   Sums = cell(tensorSize, 1);
   for x = tensorIxs
       nxSet = tensorIxs(tensorIxs ~= x);
       Sums{x} = sumTensor(invLsTensor, nxSet);                            % elementwise sums of invLsTensor
       Er{x} = reshape(eigValues{x}, size(Sums{x}))- Sums{x};              % error with respect to each marginal covariance eigenvalue
       fx{x} = reshape(Er{x}, dim(x), 1).^2;                               
   end
   f = sum(vertcat(fx{:}))./normalizeTerm;                                 % the objective value
   %% build the gradient from the blocks
   gradf_L = nan(sum(dim),1);                                              % gradient
   for x = tensorIxs
       nxSet = tensorIxs(tensorIxs ~= x);                                  
       gradfx_Lx = reshape(bsxfun(@times, 2.*Er{x},...
           sumTensor(invSquareLsTensor, nxSet)),...
           dim(x),1);
       
       gradfy_Lx = nan(dim(x), tensorSize-1);
       z = 1;
       for y = nxSet(:).'
           nxySet = tensorIxs((tensorIxs ~= x) & (tensorIxs ~= y));
           gradfy_Lx(:, z) = reshape(sumTensor(bsxfun(@times, ...
                2.*Er{y},...
                sumTensor(invSquareLsTensor, nxySet)), y),...
                dim(x),1);           
           z = z+1;
       end   
       gradf_Lx = sum([gradfx_Lx gradfy_Lx], 2);
       gradf_L(sum(dim(1:x-1))+(1:dim(x))) = gradf_Lx;
   end
   gradf_L = gradf_L./normalizeTerm;                                 
end