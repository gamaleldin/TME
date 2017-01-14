%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% <TME>
% Copyright (C) 2017 Gamaleldin F. Elsayed and John P. Cunningham 
%       (see full notice in README)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [kronSumDs] = diagKronSum(Ds)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function evaluates the efficient evaluation of kronecker (tensor) 
% sums of diagonal matrices.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
%       - Ds: is a cell where each element contains the diagonal elements
%       of each matrix.
% Outputs:
%       - kronSumLs: is the result of Dn \kronsum Dn-1 .....\kronsum D1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [kronSumDs] = diagKronSum(Ds)
tensorSize = length(Ds);
kronSumDs = 0;
for x = 1:tensorSize
    kronSumDs = kronSumDs*ones(length(Ds{x}),1)'+ones(length(kronSumDs),1)*Ds{x}.';
    kronSumDs = kronSumDs(:);
end
end