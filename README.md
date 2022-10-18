# Mammogram Classification
Persistent Homology for Breast Tumor Classification using Mammogram Scans

## Table of contents
1. [Feature extraction](#feature-extraction)
2. [Classification](#classification)


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

