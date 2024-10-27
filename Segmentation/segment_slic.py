import numpy as np
from osgeo import gdal
from skimage import exposure
from skimage.segmentation import quickshift, felzenszwalb, slic
import time

file = './images/Sentinel2A_subset_bands_BGRNir4MIR1MIR2.tif'

# Open the raster file using GDAL
driver = gdal.GetDriverByName('GTiff')
ds = gdal.Open(file)
nbands = ds.RasterCount

band_data = []
print('bands', ds.RasterCount, 'rows', ds.RasterYSize, 'columns', ds.RasterXSize)

# Read all bands into a list
for i in range(1, nbands + 1):
    band = ds.GetRasterBand(i).ReadAsArray()
    band_data.append(band)

# Stack bands into a 3D array (height, width, num_bands)
band_data = np.dstack(band_data)

# Rescale intensity
img = exposure.rescale_intensity(band_data)

# Perform segmentation using K-Means based image segmentation
seg_time = time.time()
segments_slic = slic(img, n_segments=60, compactness=0.07, sigma=1, start_label=1)
print('SLIC time: ', time.time() - seg_time)

# Print number of segments
print(f'SLIC number of segments: {len(np.unique(segments_slic))}')

# Save segments
np.save('./results/segments_slic.npy', segments_slic)