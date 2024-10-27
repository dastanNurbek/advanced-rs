import numpy as np
from osgeo import gdal
from skimage import exposure
from skimage.segmentation import quickshift
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

file = './images/Sentinel2A_subset_bands_BGRNir4MIR1MIR2.tif'
output = './results/segmented_S2.tif'

driver = gdal.GetDriverByName('GTiff')
ds = gdal.Open(file)
nbands = ds.RasterCount

band_data = []
print('bands', ds.RasterCount, 'rows', ds.RasterYSize, 'columns', ds.RasterXSize)

for i in range(1, nbands+1):
    band = ds.GetRasterBand(i).ReadAsArray()
    band_data.append(band)

band_data = np.dstack(band_data)
img = exposure.rescale_intensity(band_data)
segments = quickshift(img, convert2lab=False)

segments_ds = driver.Create(output, ds.RasterXSize, ds.RasterYSize,
                            1, gdal.GDT_Float32)

segments_ds.SetGeoTransform(ds.GetGeoTransform())
segments_ds.SetProjection(ds.GetProjectionRef())
segments_ds.GetRasterBand(1).WriteArray(segments)
segments_ds = None