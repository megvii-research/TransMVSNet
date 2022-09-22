clear all
close all
format compact
clc

% script to calculate distances have been measured for all included scans (UsedSets)
% DTU dataset pcd path
dataPath='/0_workspace/MVS/DTU_MATLAB_eval';
% my result path
plyPath='/0_workspace/MVS/DTU_MATLAB_eval/test_train_syncbn_4src_1021';
% eval result path
resultsPath='/0_workspace/MVS/DTU_MATLAB_eval/test_train_syncbn_4src_1021_result';

method_string='mvsnet';
light_string=''; % l3 is the setting with all lights on, l7 is randomly sampled between the 7 settings (index 0-6)
representation_string='Points'; %mvs representation 'Points' or 'Surfaces'

switch representation_string
    case 'Points'
        eval_string='_Eval_'; %results naming
        settings_string='';
end

% get sets used in evaluation
UsedSets=[1 4 9 10 11 12 13 15 23 24 29 32 33 34 48 49 62 75 77 110 114 118];

dst=0.2;    %Min dist between points when reducing

for cIdx=1:length(UsedSets)
    %Data set number
    cSet = UsedSets(cIdx)
    %input data name
    %DataInName=[plyPath sprintf('/%sscan%d%s%s.ply',lower(method_string),cSet,light_string,settings_string)]
    DataInName=[plyPath sprintf('/mvsnet%03d_l3.ply',cSet)]
    
    %results name
    EvalName=[resultsPath method_string eval_string num2str(cSet) '.mat']
    
    %check if file is already computed
    if(~exist(EvalName,'file'))
        disp(DataInName);
        
        time=clock;time(4:5), drawnow
        
        tic
        Mesh = plyread(DataInName);
        Qdata=[Mesh.vertex.x Mesh.vertex.y Mesh.vertex.z]';
        toc
        
        BaseEval=PointCompareMain(cSet,Qdata,dst,dataPath);
        
        disp('Saving results'), drawnow
        toc
        save(EvalName,'BaseEval');
        toc
        
        % write obj-file of evaluation
        % BaseEval2Obj_web(BaseEval,method_string, resultsPath)
        % toc
        time=clock;time(4:5), drawnow
    
        BaseEval.MaxDist=20; %outlier threshold of 20 mm
        
        BaseEval.FilteredDstl=BaseEval.Dstl(BaseEval.StlAbovePlane); %use only points that are above the plane 
        BaseEval.FilteredDstl=BaseEval.FilteredDstl(BaseEval.FilteredDstl<BaseEval.MaxDist); % discard outliers
    
        BaseEval.FilteredDdata=BaseEval.Ddata(BaseEval.DataInMask); %use only points that within mask
        BaseEval.FilteredDdata=BaseEval.FilteredDdata(BaseEval.FilteredDdata<BaseEval.MaxDist); % discard outliers
        
        fprintf("mean/median Data (acc.) %f/%f\n", mean(BaseEval.FilteredDdata), median(BaseEval.FilteredDdata));
        fprintf("mean/median Stl (comp.) %f/%f\n", mean(BaseEval.FilteredDstl), median(BaseEval.FilteredDstl));
    end
end
