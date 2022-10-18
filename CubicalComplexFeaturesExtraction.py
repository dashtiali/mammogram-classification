"""
Extracting Features
@author: Dashti
"""

import numpy as np
import cv2
from ripser import ripser
import glob
import os
import multiprocessing as mp
from gudhi import representations, CubicalComplex

# Function to compute persistent binnng features
def compute_binning(dim_n, thresholds):
    n = np.size(dim_n, 0)
    intersects = []
    
    if(n > 0):
        if(np.size(thresholds) == 1):
            thresholds = [thresholds]
            
        for threshold in thresholds:
            int_count = 0
            for i in range(n):
                if(threshold >= dim_n[i,0] and threshold <= dim_n[i,1]):
                    int_count += 1
                    
            intersects.append(int_count)
        
    return intersects

# Function to compute persistent statistics features
def get_barcode_stats(barcode):
    # Computing Statistics from Persistent Barcodes

    if (np.size(barcode) > 0):
        # Average of Birth and Death of the barcode
        bc_av0, bc_av1 = np.mean(barcode, axis=0)
        # STDev of Birth and Death of the barcode
        bc_std0, bc_std1 = np.std(barcode, axis=0)
        # Median of Birth and Death of the barcode
        bc_med0, bc_med1 = np.median(barcode, axis=0)
        diff_barcode = np.subtract([i[1] for i in barcode], [i[0] for i in barcode])
        diff_barcode = np.absolute(diff_barcode)
        # Average of the length of Bars        
        bc_lengthAverage = np.mean(diff_barcode)
        # STD of length of Bars
        bc_lengthSTD = np.std(diff_barcode)
        # Median of length of Bars
        bc_lengthMedian = np.median(diff_barcode)
        # Number of Bars
        bc_count = len(diff_barcode)

        bar_stats = np.array([bc_av0, bc_av1, bc_std0, bc_std1, bc_med0, bc_med1,
                     bc_lengthAverage, bc_lengthSTD, bc_lengthMedian, bc_count])
    else:
        bar_stats= np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
    bar_stats[~np.isfinite(bar_stats)] = 0
    
    return bar_stats

# Function to compute persistent image features
def get_pers_imgs(barcode, persIm):
    feature_vectors = []
    
    if(np.size(barcode) > 0):
        feature_vectors = persIm.fit_transform([barcode])
    
    return feature_vectors

# Function to compute persistent landscape features
def get_pers_lands(barcode, persLand):
    feature_vectors = []
    
    if(np.size(barcode) > 0):
        feature_vectors = persLand.fit_transform([barcode])
    
    return feature_vectors

# Function to create cubical complex object
def get_cubical_complex_image_object(img):
    fl = np.ndarray.flatten(img)
    nx = img.shape[0]
    ny = img.shape[1]

    cc_obj = CubicalComplex(dimensions = [nx ,ny], top_dimensional_cells = fl)

    return cc_obj

# Function to compute featurized barcodes
def get_pd_featurized_barcodes(imgPath):
    print(imgPath)
    
    dim0_bin = []
    dim0_stats = []
    dim1_bin = []
    dim1_stats = []
    dim0_persims = []
    dim1_persims = []
    dim0_perland = []
    dim1_perland = []

    # Binning range
    binRange = np.arange(0, 30)

    # Persistent Image
    persIm = representations.PersistenceImage(resolution=[30, 30])

    # Persistent Landscape
    persLand = representations.Landscape(resolution=100)
    
    img = cv2.imread(imgPath,0)

    cc_obj = get_cubical_complex_image_object(img)
    ph = cc_obj.persistence()

    dim0 = cc_obj.persistence_intervals_in_dimension(0)
    dim1 = cc_obj.persistence_intervals_in_dimension(1)

    dim0 = dim0[dim0[:, 0] != np.inf]
    dim0 = dim0[dim0[:, 1] != np.inf]

    dim1 = dim1[dim1[:, 0] != np.inf]
    dim1 = dim1[dim1[:, 1] != np.inf]
    
    dim0_bin= compute_binning(dim0, binRange)
    dim0_stats= get_barcode_stats(dim0)
    
    dim1_bin= compute_binning(dim1, binRange)
    dim1_stats= get_barcode_stats(dim1)
    
    dim0_persims= get_pers_imgs(dim0, persIm)
    dim1_persims= get_pers_imgs(dim1, persIm)
    
    dim0_perland= get_pers_lands(dim0, persLand)
    dim1_perland= get_pers_lands(dim1, persLand)

    return [os.path.basename(imgPath), dim0_bin, dim0_stats, dim1_bin, dim1_stats, dim0_persims, dim1_persims, dim0_perland, dim1_perland]

# Main thread to run the script
if __name__ == '__main__':
    
    DatabasesPath = 'Datasets\\'
    Databases = [ 'DDSM_Mass_257images',
                        'DDSM_Normal_302images',
                        'Mini_Mias_Abnormal113images',
                        'Mini_Mias_Normal209images']

    for setName in Databases:
        images = [f for f in glob.glob(DatabasesPath + setName + '\\' + "*.pgm", recursive=True)]
        
        pool = mp.Pool(10)
        results = pool.map(get_pd_featurized_barcodes, images, chunksize=1)
        pool.close()
        
        outputFolder = 'ExportedFeatures'
        if not os.path.isdir(f'{outputFolder}//{setName}'):
                os.mkdir(f'{outputFolder}//{setName}')
                
        np.savetxt(f'{outputFolder}\\{setName}\\file_names.csv', [i[0] for i in results], delimiter=",", fmt='%s')

        np.savetxt(f'{outputFolder}\\{setName}\\dim0_bin.csv', [i[1] for i in results], delimiter=",")
        np.savetxt(f'{outputFolder}\\{setName}\\dim0_stats.csv', [i[2] for i in results], delimiter=",")
        np.savetxt(f'{outputFolder}\\{setName}\\dim1_bin.csv', [i[3] for i in results], delimiter=",")
        np.savetxt(f'{outputFolder}\\{setName}\\dim1_stats.csv', [i[4] for i in results], delimiter=",")
        np.savetxt(f'{outputFolder}\\{setName}\\dim0_persims.csv', [i[5][0] for i in results], delimiter=",")
        np.savetxt(f'{outputFolder}\\{setName}\\dim1_persims.csv', [i[6][0] for i in results], delimiter=",")
        np.savetxt(f'{outputFolder}\\{setName}\\dim0_perland.csv', [i[7][0] for i in results], delimiter=",")
        np.savetxt(f'{outputFolder}\\{setName}\\dim1_perland.csv', [i[8][0] for i in results], delimiter=",")
