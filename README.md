# Mammogram Classification
Persistent Homology for Breast Tumor Classification using Mammogram Scans

## Table of contents
1. [Feature extraction](#feature-extraction)
2. [Classification](#classification)
3. [How to use the code effectively](#how-to-use-the-code-effectively)


## Feature extraction

| Name | Description  |
|----------------------------------------------------------------------------------------------------------|----------------------------------|
|[ULBPFeaturesExtraction.py](https://github.com/dashtiali/mammogram-classification/blob/main/ULBPFeaturesExtraction.py) | script to compute ULBP domain PH featurized barcodes |
|[ULBPExportFeaturesToCSV.py](https://github.com/dashtiali/mammogram-classification/blob/main/ULBPExportFeaturesToCSV.py) | script to concatenate all featurized barcode of each rotation of an ULBP geometry into one feature vector and export to csv |
|[CubicalComplexFeaturesExtraction.py](https://github.com/dashtiali/mammogram-classification/blob/main/CubicalComplexFeaturesExtraction.py) | script to compute cubical complex PH featurized barcodes |

## Classification
| Name | Description  |
|----------------------------------------------------------------------------------------------------------|----------------------------------|
|[PH_for_Mammogram_Classification_SVM.m](https://github.com/dashtiali/mammogram-classification/blob/main/PH_for_Mammogram_Classification_SVM.m) | Matlab script to perform classification using 'Binary_SVM_optimised' function |
|[Binary_SVM_optimised.m](https://github.com/dashtiali/mammogram-classification/blob/main/Binary_SVM_optimised.m) | Matlab script of Binary_SVM_optimised function |

## How to use the code effectively:

1. The function 'ULBPFeaturesExtraction.py ' extracts ULBP landmarks and computes persistence barcodes based on Vietoris-Rips simplicial complex filtration. It also vectorise the space of persistent barcodes using 4 techniques of persistence Binning, Persistence Landscapes, Persistence Image and Persistence Statistics. This function finally saves 59 vectorised Persistence barcodes ,as numpy array, see Figure 2 from the paper.

2. The Function 'ULBPExportFeaturesToCSV.py ' concatenate all featurized barcode of each rotation of ULBP into 7 groups according to their ULBP geometries, see Figure 2 from the paper. It then saves it as a .csv files ready for classificcation stage.

3. Finally, The classification stage is performed in MATLAB ( Version R2021b) using ' PH_for_Mammogram_Classification_SVM.m' function. It reads featurised barcodes prepared by 'ULBPExportFeaturesToCSV.py ' function and performs the classification using optimised SVM via ' Binary_SVM_optimised.m' function. 'Binary_SVM_optimised.m' function performs 5-fold-cross-validation in balanced manner and optimises all hyperparameters of SVM and outputs best kernel and confusion matrix for each of the fold.

