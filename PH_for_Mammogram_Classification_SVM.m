
%%%%% Reading Topological Feature Vectors from both classes
%%%% We read four types of the featurisation methods of both classes of the
%%%% same dataset (MM=MiniMIAS, DDSM)
x1=csvread('MM_abnormal_dim0_persims.csv');
x2=csvread('MM_abnormal_dim1_persims.csv');
x3=csvread('MM_abnormal_dim0_perland.csv');
x4=csvread('MM_abnormal_dim1_perland.csv');
% 
x5=csvread('MM_normal_dim0_persims.csv');
x6=csvread('MM_normal_dim1_persims.csv');
x7=csvread('MM_normal_dim0_perland.csv');
x8=csvread('MM_normal_dim1_perland.csv');
% 
x11=csvread('MM_abnormal_dim0_ints.csv');
x22=csvread('MM_abnormal_dim1_ints.csv');
x33=csvread('MM_abnormal_dim0_stats.csv');
x44=csvread('MM_abnormal_dim1_stats.csv');

x55=csvread('MM_normal_dim0_ints.csv');
x66=csvread('MM_normal_dim1_ints.csv');
x77=csvread('MM_normal_dim0_stats.csv');
x88=csvread('MM_normal_dim1_stats.csv');

%%
%%%%% SVM Classification stage %%%%%%%%%%%%
%%%%% The classification is conducted using 'Binary_SVM_optimised' function
%%%%% that optimises all hyperparameters of SVM, adds labels to the two classes
%%%%% and outputs kernel type, AUC and confusion matrix. 
%%%%% 

%%%%% PersImage Classification %%%%%%
[kernel_dim0_PersImg, dim0_PersImg_SVM_AUC_sets, dim0_PersImg_SVM_ConfMats_sets]=Binary_SVM_optimised(x5,x1);
% [~, dim0_PersImg_Ensemble_AUC_sets, dim0_PersImg_Ensemble_confuMatrices]=Binary_Ensemble_optimised(x5,x1);
% 
[kernel_dim1_PersImg, dim1_PersImg_SVM_AUC_sets, dim1_PersImg_SVM_ConfMats_sets]=Binary_SVM_optimised(x6,x2);
% [~, dim1_PersImg_Ensemble_AUC_sets, dim1_PersImg_Ensemble_confuMatrices]=Binary_Ensemble_optimised(x6,x2);
% 
[kernel_dim01_PersImg, dim01_PersImg_SVM_AUC_sets, dim01_PersImg_SVM_ConfMats_sets]=Binary_SVM_optimised([x5,x6],[x1,x2]);
% [~, dim01_PersImg_Ensemble_AUC_sets, dim01_PersImg_Ensemble_confuMatrices]=Binary_Ensemble_optimised([x5,x6],[x1,x2]);
% 
% %%%%% PersLandscape SVM %%%%%%%%%
[kernel_dim0_PersLAnd, dim0_PersLand_SVM_AUC_sets, dim0_PersLand_SVM_ConfMats_sets]=Binary_SVM_optimised(x7,x3);
% [~, dim0_PersLand_Ensemble_AUC_sets, dim0_PersLand_Ensemble_confuMatrices]=Binary_Ensemble_optimised(x7,x3);
% 
[kernel_dim1_PersLand, dim1_PersLand_SVM_AUC_sets, dim1_PersLand_SVM_ConfMats_sets]=Binary_SVM_optimised(x8,x4);
% [~, dim1_PersLand_Ensemble_AUC_sets, dim1_PersLand_Ensemble_confuMatrices]=Binary_Ensemble_optimised(x8,x4);
% 
[kernel_dim01_PersLand, dim01_PersLand_SVM_AUC_sets, dim01_PersLand_SVM_ConfMats_sets]=Binary_SVM_optimised([x7,x8],[x3,x4]);
% [~, dim01_PersLand_Ensemble_AUC_sets, dim01_PersLand_Ensemble_confuMatrices]=Binary_Ensemble_optimised([x7,x8],[x3,x4]);
% 
% %%%%% Binning SVM 
[kernel_dim0_bin, dim0_Bining_SVM_AUC_sets, dim0_Bining_SVM_ConfMats_sets]=Binary_SVM_optimised(x55,x11);
% [~, dim0_Bining_Ensemble_AUC_sets, dim0_Bining_Ensemble_confuMatrices]=Binary_Ensemble_optimised(x55,x11);
% 
[kernel_dim1_bin, dim1_Bining_SVM_AUC_sets, dim1_Bining_SVM_ConfMats_sets]=Binary_SVM_optimised(x66,x22);
% [~, dim1_Bining_Ensemble_AUC_sets, dim1_Bining_Ensemble_confuMatrices]=Binary_Ensemble_optimised(x66,x22);
% 
[kernel_dim01_bin, dim01_Bining_SVM_AUC_sets, dim01_Bining_SVM_ConfMats_sets]=Binary_SVM_optimised([x55,x66],[x11,x22]);
% [~, dim01_Bining_Ensemble_AUC_sets, dim01_Bining_Ensemble_confuMatrices]=Binary_Ensemble_optimised([x55,x66],[x11,x22]);
% 
% %%%%%%% PersStatistics SVM
[kernel_dim0_stat, dim0_Stats_SVM_AUC_sets, dim0_Stats_SVM_ConfMats_sets]=Binary_SVM_optimised(x77,x33);
% [~, dim0_Stats_Ensemble_AUC_sets, dim0_Stats_Ensemble_confuMatrices]=Binary_Ensemble_optimised(x77,x33);
% 
[kernel_dim1_stat, dim1_Stats_SVM_AUC_sets, dim1_Stats_SVM_ConfMats_sets]=Binary_SVM_optimised(x88,x44);
% [~, dim1_Stats_Ensemble_AUC_sets, dim1_Stats_Ensemble_confuMatrices]=Binary_Ensemble_optimised(x88,x44);
% 
[kernel_dim01_stat, dim01_Stats_SVM_AUC_sets, dim01_Stats_SVM_ConfMats_sets]=Binary_SVM_optimised([x77,x88],[x33,x44]);
% [~, dim01_Stats_Ensemble_AUC_sets, dim01_Stats_Ensemble_confuMatrices]=Binary_Ensemble_optimised([x77,x88],[x33,x44]);

%%%%% storing kernels of each featurisation map ( comment if not needed)
kernels=[kernel_dim0_bin;kernel_dim1_bin;kernel_dim01_bin;...
    kernel_dim0_stat;kernel_dim1_stat;kernel_dim01_stat;...
    kernel_dim0_PersImg;kernel_dim1_PersImg;kernel_dim01_PersImg;...
    kernel_dim0_PersLAnd;kernel_dim1_PersLand;kernel_dim01_PersLand
    ];
%%%%% Storing AUC numbers if needed, for PH dimensions of 0,1 and 0+1
AUCs=[dim0_Bining_SVM_AUC_sets;dim1_Bining_SVM_AUC_sets;dim01_Bining_SVM_AUC_sets;...
    dim0_Stats_SVM_AUC_sets;dim1_Stats_SVM_AUC_sets;dim01_Stats_SVM_AUC_sets;...
    dim0_PersImg_SVM_AUC_sets;dim1_PersImg_SVM_AUC_sets;dim01_PersImg_SVM_AUC_sets;...
    dim0_PersLand_SVM_AUC_sets;dim1_PersLand_SVM_AUC_sets;dim01_PersLand_SVM_AUC_sets
    ];

%%%%% Storing Conf-matrices for all featuremaps in dimension 0,1 and 0+1
ConfMats=[dim0_Bining_SVM_ConfMats_sets;dim1_Bining_SVM_ConfMats_sets;dim01_Bining_SVM_ConfMats_sets;...
    dim0_Stats_SVM_ConfMats_sets;dim1_Stats_SVM_ConfMats_sets;dim01_Stats_SVM_ConfMats_sets;...
    dim0_PersImg_SVM_ConfMats_sets;dim1_PersImg_SVM_ConfMats_sets;dim01_PersImg_SVM_ConfMats_sets;...
    dim0_PersLand_SVM_ConfMats_sets;dim1_PersLand_SVM_ConfMats_sets;dim01_PersLand_SVM_ConfMats_sets
    ];
%%%% Finally, we calculate and analyse Sensitivity, specificity, Accuracy
%%%% and other analyses in Excel file externally. 