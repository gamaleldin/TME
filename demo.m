%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% <TME>
% Copyright (C) 2017 Gamaleldin F. Elsayed and John P. Cunningham 
%       (see full notice in README)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a demonstration of how to use this code package 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

startup
rng('shuffle', 'twister') % randomize the seed

surrogate_type = 'surrogate-TNC';
% this data is a preprocessed version of the data available online from 
% http://stat.columbia.edu/~cunningham/pdf/ChurchlandNature2012_code.zip
load exampleData.mat 
%% quantify the linear dynmaical structure of original data by a summary statistic (R2)
model_dim = 10;
times_msk = t>-50 & t<350; % select movement-related times
[R2_data] = summarizeLDS(dataTensor(times_msk, :, :), model_dim, false); % function that evaluates the summary statistic of the LDS structure
%% quantify primary features of the original data
[targetSigmaT, targetSigmaN, targetSigmaC, M] = extractFeatures(dataTensor);
%% sample many surrogates and build null distribution of summary statistics
numSurrogates = 100;
params = [];


if strcmp(surrogate_type, 'surrogate-T')
    params.margCov{1} = targetSigmaT;
    params.margCov{2} = [];
    params.margCov{3} = [];
    params.meanTensor = M.T;
elseif strcmp(surrogate_type, 'surrogate-TN')
    params.margCov{1} = targetSigmaT;
    params.margCov{2} = targetSigmaN;
    params.margCov{3} = [];
    params.meanTensor = M.TN;
elseif strcmp(surrogate_type, 'surrogate-TNC')
    params.margCov{1} = targetSigmaT;
    params.margCov{2} = targetSigmaN;
    params.margCov{3} = targetSigmaC;
    params.meanTensor = M.TNC; 
else
    error('please specify a correct surrogate type') 
end

maxEntropy = fitMaxEntropy(params);             % fit the maximum entropy distribution
R2_surr = nan(numSurrogates, 1);
for i = 1:numSurrogates
    fprintf('surrogate %d from %d \n', i, numSurrogates)
    [surrTensor] = sampleTME(maxEntropy);       % generate TME random surrogate data.
    [R2_surr(i)] = summarizeLDS(surrTensor(times_msk, :, :), model_dim, false);
end
%% evaluate a P value
P = mean(R2_data<= R2_surr); % (upper-tail test)

if P>=0.05
   fprintf('P value = %1.0e\n', P)
else
   fprintf('P value < %.3f\n', (P<0.001)*0.001 + (P<0.01 & P>=0.001)*0.01 + (P<0.05 & P>=0.01)*0.05)
end
%%%%%%%%%%%%%%%% plot null distribution
x = 0:0.03:1;
h = hist(R2_surr, x);
hf=figure;
set(hf, 'color', [1 1 1]);
hold on
box on
hb = bar(x, h);
set(hb,'facecolor',[0.5000    0.3118    0.0176],'barwidth',1,'edgecolor','none')
p = plot(R2_data, 0, 'ko', 'markerfacecolor', 'k', 'markersize',10);
xlabel('summary statistic (R^2)')
ylabel('count')
xlim([0 1])
set(gca, 'FontSize',12)
set(gca, 'xtick',[-1 0 1])
legend([p, hb], {'original data', surrogate_type})
legend boxoff

