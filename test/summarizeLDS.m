function [R2] = summarizeLDS(dataTensor, model_dim, crossValFlg)
[T, N, C] = size(dataTensor);
XN = reshape(permute(dataTensor,[1 3 2]), [], N);
%% now do traditional PCA
meanXN= mean(XN);
[PCvectors, ~, ~] = pca(XN);  % apply PCA to the analyzed times
maskT1Orig = true(T,1);maskT1Orig(end) = false;maskT1Orig = repmat(maskT1Orig, C, 1);
maskT2Orig = true(T,1);maskT2Orig(1) = false;maskT2Orig = repmat(maskT2Orig, C,1);
R2 = nan(length(model_dim), 1);
if ~crossValFlg
    for i = 1:length(model_dim)
        PCvectors_i = PCvectors(:,1:model_dim(i));
        XNred = bsxfun(@minus, XN, meanXN)*PCvectors_i;
        dState = (XNred(maskT2Orig,:) - XNred(maskT1Orig,:));  % the masks just give us earlier and later times within each condition
        preState = XNred(maskT1Orig,:);  % just for convenience, keep the earlier time in its own variable
        M = ((dState)'/preState');  % M takes the state and provides a fit to dState
        fitErrorM = dState'- M*preState';
        varDState = sum(dState(:).^2);  % original data variance
        R2(i) = (varDState - sum(fitErrorM(:).^2)) / varDState;  % how much is explained by the overall fit via M
    end
%%
else

for i = 1:length(model_dim)
    fitErrorMTest =[];
    dStateTest = [];
    for c = 1:C
        maskTrain = true(1,C);maskTrain(c) = false; maskTrain = reshape(repmat(maskTrain, T, 1), [],1);
        XNTrain = XN(maskTrain, :);
        XNredTrain = bsxfun(@minus, XNTrain, meanXN)*PCvectors(:, 1:model_dim(i));
        
        maskT1Train = true(T,1);maskT1Train(end) = false;maskT1Train = repmat(maskT1Train, C-1, 1);
        maskT2Train = true(T,1);maskT2Train(1) = false;maskT2Train = repmat(maskT2Train, C-1,1);

        dStateTrain = (XNredTrain(maskT2Train,:) - XNredTrain(maskT1Train,:));  % the masks just give us earlier and later times within each condition
        preStateTrain = XNredTrain(maskT1Train,:);  % just for convenience, keep the earlier time in its own variable
        M = ((dStateTrain)'/preStateTrain');  % M takes the state and provides a fit to dState

        %% Test
        maskTest = ~maskTrain;
        XNTest = XN(maskTest,:);
        XNredTest = bsxfun(@minus, XNTest, meanXN)*PCvectors(:, 1:model_dim(i));
        
        maskT1Test = true(T,1);maskT1Test(end) = false;
        maskT2Test = true(T,1);maskT2Test(1) = false;
        
        dStateTest_c = (XNredTest(maskT2Test,:) - XNredTest(maskT1Test,:));
        preStateTest_c = XNredTest(maskT1Test,:);  % just for convenience, keep the earlier time in its own variable
        dStateTest = [dStateTest dStateTest_c];  % the masks just give us earlier and later times within each condition
        fitErrorMTest = [fitErrorMTest; (dStateTest_c'- M*preStateTest_c')];
    end
    varDStateTest_d = sum(dStateTest(:).^2);  % original data variance
    R2(i) = (varDStateTest_d - sum(fitErrorMTest(:).^2)) / varDStateTest_d;  % how much is explained by the overall fit via M
end
end
end


