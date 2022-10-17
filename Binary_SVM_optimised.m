
function [Kernel_type, AUC_sets, confuMatrices]=Binary_SVM_optimised(Normal_Feature,Abnormal_Features)
%%%% This function perfurmes 5 fold
Fold_Num=5;
Normal_size=size(Normal_Feature,1);
Abnormal_size=size(Abnormal_Features,1);
for i=1:1
    rng(i)
r1 = randperm(Normal_size);r2 = randperm(Abnormal_size);
Normal_Feature=Normal_Feature(r1,:);
Abnormal_Features=Abnormal_Features(r2,:);
normal_par=floor(Normal_size/Fold_Num); 
Abnormal_par=floor(Abnormal_size/Fold_Num);

p1_nor=Normal_Feature(1:normal_par,:);
p2_nor=Normal_Feature(normal_par+1:2*normal_par,:);
p3_nor=Normal_Feature((2*normal_par)+1:(3*normal_par),:);
p4_nor=Normal_Feature((3*normal_par)+1:(4*normal_par),:);
p5_nor=Normal_Feature((4*normal_par)+1:Normal_size,:);
p1_abn=Abnormal_Features(1:Abnormal_par,:);
p2_abn=Abnormal_Features(Abnormal_par+1:2*Abnormal_par,:);
p3_abn=Abnormal_Features((2*Abnormal_par)+1:(3*Abnormal_par),:);
p4_abn=Abnormal_Features((3*Abnormal_par)+1:(4*Abnormal_par),:);
p5_abn=Abnormal_Features((4*Abnormal_par)+1:Abnormal_size,:);

training_data1=[[p2_nor;p3_nor;p4_nor;p5_nor],zeros(size([p2_nor;p3_nor;p4_nor;p5_nor],1),1);[p2_abn;p3_abn;p4_abn;p5_abn],ones(size([p2_abn;p3_abn;p4_abn;p5_abn],1),1)];
training_data2=[[p1_nor;p3_nor;p4_nor;p5_nor],zeros(size([p1_nor;p3_nor;p4_nor;p5_nor],1),1);[p1_abn;p3_abn;p4_abn;p5_abn],ones(size([p1_abn;p3_abn;p4_abn;p5_abn],1),1)];
training_data3=[[p1_nor;p2_nor;p4_nor;p5_nor],zeros(size([p1_nor;p2_nor;p4_nor;p5_nor],1),1);[p1_abn;p2_abn;p4_abn;p5_abn],ones(size([p1_abn;p2_abn;p4_abn;p5_abn],1),1)];
training_data4=[[p1_nor;p2_nor;p3_nor;p5_nor],zeros(size([p1_nor;p2_nor;p3_nor;p5_nor],1),1);[p1_abn;p2_abn;p3_abn;p5_abn],ones(size([p1_abn;p2_abn;p3_abn;p5_abn],1),1)];
training_data5=[[p1_nor;p2_nor;p3_nor;p4_nor],zeros(size([p1_nor;p2_nor;p3_nor;p4_nor],1),1);[p1_abn;p2_abn;p3_abn;p4_abn],ones(size([p1_abn;p2_abn;p3_abn;p4_abn],1),1)];

testing_data1=[p1_nor;p1_abn];
testing_data2=[p2_nor;p2_abn];
testing_data3=[p3_nor;p3_abn];
testing_data4=[p4_nor;p4_abn];
testing_data5=[p5_nor;p5_abn];

SVMModel1 = fitcsvm(training_data1(:,1:end-1),training_data1(:,end),'Standardize',true,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('Verbose',0,'ShowPlots',0));
SVMModel2 = fitcsvm(training_data2(:,1:end-1),training_data2(:,end),'Standardize',true,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('Verbose',0,'ShowPlots',0));
SVMModel3 = fitcsvm(training_data3(:,1:end-1),training_data3(:,end),'Standardize',true,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('Verbose',0,'ShowPlots',0));
SVMModel4 = fitcsvm(training_data4(:,1:end-1),training_data4(:,end),'Standardize',true,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('Verbose',0,'ShowPlots',0));
SVMModel5 = fitcsvm(training_data5(:,1:end-1),training_data5(:,end),'Standardize',true,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('Verbose',0,'ShowPlots',0));

[fold1_label,fold1_score] = predict(SVMModel1,testing_data1);
[fold2_label,fold2_score] = predict(SVMModel2,testing_data2);
[fold3_label,fold3_score] = predict(SVMModel3,testing_data3);
[fold4_label,fold4_score] = predict(SVMModel4,testing_data4);
[fold5_label,fold5_score] = predict(SVMModel5,testing_data5);
C1 = confusionmat(fold1_label,[zeros(size(p1_nor,1),1);ones(size(p1_abn,1),1)]);
C2 = confusionmat(fold2_label,[zeros(size(p2_nor,1),1);ones(size(p2_abn,1),1)]);
C3 = confusionmat(fold3_label,[zeros(size(p3_nor,1),1);ones(size(p3_abn,1),1)]);
C4 = confusionmat(fold4_label,[zeros(size(p4_nor,1),1);ones(size(p4_abn,1),1)]);
C5 = confusionmat(fold5_label,[zeros(size(p5_nor,1),1);ones(size(p5_abn,1),1)]);

[X1,Y1,T1,AUC1,OPTROCPT1] = perfcurve([zeros(size(p1_nor,1),1);ones(size(p1_abn,1),1)],fold1_score(:,1),0);
[X2,Y2,T2,AUC2,OPTROCPT2] = perfcurve([zeros(size(p2_nor,1),1);ones(size(p2_abn,1),1)],fold2_score(:,1),0);
[X3,Y3,T3,AUC3,OPTROCPT3] = perfcurve([zeros(size(p3_nor,1),1);ones(size(p3_abn,1),1)],fold3_score(:,1),0);
[X4,Y4,T4,AUC4,OPTROCPT4] = perfcurve([zeros(size(p4_nor,1),1);ones(size(p4_abn,1),1)],fold4_score(:,1),0);
[X5,Y5,T5,AUC5,OPTROCPT5] = perfcurve([zeros(size(p5_nor,1),1);ones(size(p5_abn,1),1)],fold5_score(:,1),0);

AUC_sets(i,:)=[AUC1,AUC2,AUC3,AUC4,AUC5];
confuMatrices=[C1,C2,C3,C4,C5];
Kernel_type={SVMModel1.ModelParameters.KernelFunction,...
                SVMModel2.ModelParameters.KernelFunction,...
                SVMModel3.ModelParameters.KernelFunction,...
                SVMModel4.ModelParameters.KernelFunction,...
                SVMModel5.ModelParameters.KernelFunction};
end

end