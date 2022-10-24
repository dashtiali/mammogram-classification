import numpy as np
import os

def numpy_fillna(data, maxlen):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(maxlen) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape)
    out[mask] = np.concatenate(data)
    return out

def fillEmpty(inputarray):
    for i in range(len(inputarray)):
        if(len(inputarray[i]) < 1):
            inputarray[i] = np.zeros((1,len(inputarray[0][0])))
    return inputarray


# This function will concatenate all featurized barcode of each rotation of
# a ULBP geometry into one feature vector and export to csv
def SaveFeaturesToCSV(datasetPath, dataset):
    for setName in dataset:
        setPath = datasetPath + setName
        outputPath = setPath + '\\CSV'

        if not os.path.isdir(f'{setPath}//CSV'):
                os.mkdir(f'{setPath}//CSV')
                
        imageNames = np.load(f'{setPath}\\file_names.npy', allow_pickle=True)
        np.savetxt(f'{outputPath}\\file_names.csv', imageNames, delimiter=",", fmt='%s')
        
        for geo in range(7):
            if not os.path.isdir(f'{outputPath}//G{geo}'):
                os.mkdir(f'{outputPath}//G{geo}')

            dim0_bin_WG = []
            dim0_stats_WG = []
            dim1_bin_WG = []
            dim1_stats_WG = []
            dim0_persims_WG = []
            dim1_persims_WG = []
            dim0_perland_WG = []
            dim1_perland_WG = []
            
            for rot in range(8):
                print(rot)
                dim0_bin = np.load(f'{setPath}\\G{geo}\\R{rot}\\dim0_bin.npy', allow_pickle=True)
                # dim0_bin = dim0_bin[:,0,:]
                dim0_bin = numpy_fillna(dim0_bin, len(dim0_bin[0]))
                
                dim0_stats = np.load(f'{setPath}\\G{geo}\\R{rot}\\dim0_stats.npy', allow_pickle=True)
                # dim0_stats = dim0_stats[:,0,:]
                dim0_stats = numpy_fillna(dim0_stats, len(dim0_stats[0]))
                
                dim1_bin = np.load(f'{setPath}\\G{geo}\\R{rot}\\dim1_bin.npy', allow_pickle=True)
                # dim1_bin = dim1_bin[:,0,:]
                dim1_bin = numpy_fillna(dim1_bin, len(dim1_bin[0]))
                
                dim1_stats = np.load(f'{setPath}\\G{geo}\\R{rot}\\dim1_stats.npy', allow_pickle=True)
                # dim1_stats = dim1_stats[:,0,:]
                dim1_stats = numpy_fillna(dim1_stats, len(dim1_stats[0]))
        
                dim0_persims = np.load(f'{setPath}\\G{geo}\\R{rot}\\dim0_persims.npy', allow_pickle=True)
                # dim0_persims = dim0_persims[:,0,:]
                dim0_persims = fillEmpty(dim0_persims)
                dim0_persims = np.concatenate(dim0_persims)
                # dim0_persims = numpy_fillna(dim0_persims,  len(dim0_persims[0]))
                
                dim1_persims = np.load(f'{setPath}\\G{geo}\\R{rot}\\dim1_persims.npy', allow_pickle=True)
                # dim1_persims = dim1_persims[:,0,:]
                dim1_persims = fillEmpty(dim1_persims)
                dim1_persims = np.concatenate(dim1_persims)
                #dim1_persims = numpy_fillna([i for i in dim1_persims],  len(dim1_persims[0]))
        
                dim0_perland = np.load(f'{setPath}\\G{geo}\\R{rot}\\dim0_perland.npy', allow_pickle=True)
                # dim0_perland = dim0_perland[:,0,:]
                dim0_perland = fillEmpty(dim0_perland)
                dim0_perland = np.concatenate(dim0_perland)
                # dim0_perland = numpy_fillna([i[0] for i in dim0_perland],  len(dim0_perland[0]))
        
                dim1_perland = np.load(f'{setPath}\\G{geo}\\R{rot}\\dim1_perland.npy', allow_pickle=True)
                # dim1_perland = dim1_perland[:,0,:]
                dim1_perland = fillEmpty(dim1_perland)
                dim1_perland = np.concatenate(dim1_perland)
                # dim1_perland = numpy_fillna([i[0] for i in dim1_perland],  len(dim1_perland[0]))
                
                if(rot == 0):
                    dim0_bin_WG = dim0_bin
                    dim0_stats_WG = dim0_stats
                    dim1_bin_WG = dim1_bin
                    dim1_stats_WG = dim1_stats
                    dim0_persims_WG = dim0_persims
                    dim1_persims_WG = dim1_persims
                    dim0_perland_WG = dim0_perland
                    dim1_perland_WG = dim1_perland
                else:
                    dim0_bin_WG = np.concatenate((dim0_bin_WG, dim0_bin), axis=1)
                    dim0_stats_WG = np.concatenate((dim0_stats_WG, dim0_stats), axis=1)
                    dim1_bin_WG = np.concatenate((dim1_bin_WG, dim1_bin), axis=1)
                    dim1_stats_WG = np.concatenate((dim1_stats_WG, dim1_stats), axis=1)
                    dim0_persims_WG = np.concatenate((dim0_persims_WG, dim0_persims), axis=1)
                    dim1_persims_WG = np.concatenate((dim1_persims_WG, dim1_persims), axis=1)
                    dim0_perland_WG = np.concatenate((dim0_perland_WG, dim0_perland), axis=1)
                    dim1_perland_WG = np.concatenate((dim1_perland_WG, dim1_perland), axis=1)
            
            np.savetxt(f'{outputPath}\\G{geo}\\dim0_bin.csv', dim0_bin_WG, delimiter=",")
            np.savetxt(f'{outputPath}\\G{geo}\\dim0_stats.csv', dim0_stats_WG, delimiter=",")
            np.savetxt(f'{outputPath}\\G{geo}\\dim1_bin.csv', dim1_bin_WG, delimiter=",")
            np.savetxt(f'{outputPath}\\G{geo}\\dim1_stats.csv', dim1_stats_WG, delimiter=",")
            np.savetxt(f'{outputPath}\\G{geo}\\dim0_persims.csv', dim0_persims_WG, delimiter=",")
            np.savetxt(f'{outputPath}\\G{geo}\\dim1_persims.csv', dim1_persims_WG, delimiter=",")
            np.savetxt(f'{outputPath}\\G{geo}\\dim0_perland.csv', dim0_perland_WG, delimiter=",")
            np.savetxt(f'{outputPath}\\G{geo}\\dim1_perland.csv', dim1_perland_WG, delimiter=",")


mainPath = 'ExportedFeatures\\'
datasets = ['DDSM_Mass_257images',
            'DDSM_Normal_302images',
            'Mini_Mias_Abnormal113images',
            'Mini_Mias_Normal209images']

SaveFeaturesToCSV(mainPath, datasets)