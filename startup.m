%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% <TME>
% Copyright (C) 2017 Gamaleldin F. Elsayed and John P. Cunningham 
% (see full notice in README)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a startup file which initializes Matlab's settings to
% allow for better use of this package.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
path(sprintf('%s/genTME', pwd), path) % Adds the subdirectories of the root folder to the path, allowing us to call functions from them.
path(sprintf('%s/util', pwd), path) % Adds the subdirectories of the root folder to the path, allowing us to call functions from them.
path(sprintf('%s/util/lbfgsb', pwd), path) % Adds the subdirectories of the root folder to the path, allowing us to call functions from them.
path(sprintf('%s/test', pwd), path) % Adds the subdirectories of the root folder to the path, allowing us to call functions from them.
path(sprintf('%s/data', pwd), path) % Adds the subdirectories of the root folder to the path, allowing us to call functions from them.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This simply clears all variables and closes all windows opened by Matlab.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
clear all
