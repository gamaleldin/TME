function [sigma_T, sigma_N, sigma_C, M, mu] = extractFeatures(dataTensor, meanTensor)
T = size(dataTensor,1);
N = size(dataTensor,2);
C = size(dataTensor,3);
M = struct('T', [], 'TN', [], 'TNC', []);
meanT = sumTensor(dataTensor, [2 3])/(C*N);
meanN = sumTensor(dataTensor, [1 3])/(C*T);
meanC = sumTensor(dataTensor, [1 2])/(T*N);
mu.T = meanT(:);
mu.N = meanN(:);
mu.C= meanC(:);
if ~exist('meanTensor','var')
%% Mean is calculated based on least norm
%% least norm mean
% % % % H = [[kron(kron(ones(C,1).', ones(N,1).'),    T*eye(T))];
% % % %      [kron(kron(ones(C,1).', N*eye(N)   ), ones(T,1).')];
% % % %      [kron(kron(C*eye(C)   , ones(N,1).'), ones(T,1).')]]./(T*N*C); 
% % % % HH = [[ T*eye(T)    ones(T,N)     ones(T,C)];
% % % %       [ones(N,T)     N*eye(N)     ones(N,C)];
% % % %       [ones(C,T)    ones(C,N)      C*eye(C)]]./(T*N*C); 
% % % % 
% % % % M.T = reshape(H.'*(HH\[mu.T; mean(mu.N)*ones(N,1); mean(mu.C)*ones(C,1)]), T, N, C);
% % % % M.TN = reshape(H.'*(HH\[mu.T; mu.N; mean(mu.C)*ones(C,1)]), T, N, C);
% % % % M.TNC = reshape(H.'*(HH\[mu.T;mu.N;mu.C]), T, N, C);
% % % % 
% % % % meanTensor = M.TNC;

%% Mean is caluclated by subtracting each reshaping mean
dataTensor0 = dataTensor;
meanT = sumTensor(dataTensor0, [2 3])/(C*N);
dataTensor0 = bsxfun(@minus, dataTensor0, meanT);
meanN = sumTensor(dataTensor0, [1 3])/(C*T);
dataTensor0 = bsxfun(@minus, dataTensor0, meanN);
meanC = sumTensor(dataTensor0, [1 2])/(T*N);
dataTensor0 = bsxfun(@minus, dataTensor0, meanC);
M.TNC = dataTensor-dataTensor0;
M.TN = repmat(sumTensor(M.TNC, [3])/(C), 1, 1, C);
M.T = repmat(sumTensor(M.TNC, [2 3])/(N*C), 1, N, C);
meanTensor = M.TNC;
%% mean is calculated based on averaging across each coordinate (i.e. collapsing one coordinate of the tensor gives a zero matrix)
% % % M = dataTensor;
% % % meanT = mean(M,1);
% % % M = bsxfun(@minus, M, meanT);
% % % meanN = mean(M,2);
% % % M = bsxfun(@minus, M, meanN);
% % % meanC = mean(M,3);
% % % M = bsxfun(@minus, M, meanC);
% % % M = dataTensor-M;
end

%% subtract the mean tensor and calculate the covariances

XT = reshape(permute((dataTensor-meanTensor),[3 2 1]), [], T);

XN = reshape(permute((dataTensor-meanTensor),[1 3 2]), [], N);

XC = reshape(permute((dataTensor-meanTensor),[1 2 3]), [], C);
sigma_T = (XT'*XT);
sigma_N = (XN'*XN);
sigma_C = (XC'*XC);
end