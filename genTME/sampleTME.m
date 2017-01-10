%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% <TME>
% Copyright (C) 2017 Gamaleldin F. Elsayed and John P. Cunningham 
%       (see full notice in README)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [surrTensors] = sampleTME(maxEntropy, numSurrogates)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function generates tensor samples from the maximum entropy 
% distribution with marginal mean and covariance constraints 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
%       - maxEntropy: tensor maximum entropy distribution parameters.
%       - numSurrogates: number of surrogate tensors
% Outputs:
%       - surrTensors: random surrogate tensors sampled from the maximum
%       entropy distribution. Different tensor samples are along the last
%       dimension of surrTensors.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [surrTensors] = sampleTME(maxEntropy, numSurrogates)
if ~exist('numSurrogates', 'var')
    numSurrogates = 1;
end
%% caculates the eigenvalues of maximum entropy covariance matrix from the lagrangians
Lagrangians = maxEntropy.Lagrangians;
eigVectors = maxEntropy.eigVectors;
dim = nan(length(eigVectors),1);
for i = 1:length(eigVectors)
    dim(i) = length(eigVectors{i});
end
tensorSize = length(dim);                                           % tensor size; ie the number of different modes of the tensor
D = 1./diagKronSum(Lagrangians);                                           
meanTensor = maxEntropy.meanTensor;
%% sample from maximum entropy distribution
x = randn(prod(dim),numSurrogates);                                           % draw random samples from a normal distribution
x = bsxfun(@times, D.^0.5, x);                               % multiply the samples by the eigenvalues of the covariance matrix the maximum entropy distribution
Qs = cell(tensorSize, 1);                                                  % load the eigenvectors of the covariance matrix of the maximum entropy distribution           
for i = 1:tensorSize
    Qs{i} = eigVectors{i};
end
x = real(kron_mvprod(Qs, x));                                              % efficiently multiply the samples by the eigenvectors of the covariance matrix of the maximum entropy distribution           
x = x + repmat(meanTensor(:), 1, numSurrogates);                                                      % add mean tensor
surrTensors = reshape(x, [dim(:); numSurrogates].');                 %surrogate tensors 
end