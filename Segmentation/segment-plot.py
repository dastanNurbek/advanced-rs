import numpy as np
from osgeo import gdal
from skimage import exposure
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

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

# Get segment data
segments = np.load('./results/segments_quick.npy')

# Create RGB image from specified bands
# Band 3 (Red), Band 2 (Green), Band 1 (Blue)
# Ensure bands are correctly indexed (1-based to 0-based indexing)
rgb_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)

# Assign bands to RGB channels
rgb_image[..., 0] = band_data[..., 2]  # Band 3 -> Red
rgb_image[..., 1] = band_data[..., 1]  # Band 2 -> Green
rgb_image[..., 2] = band_data[..., 0]  # Band 1 -> Blue

# Brightening the RGB image
# Option 1: Rescale intensities of rgb_image to brighten it
rgb_image = exposure.rescale_intensity(rgb_image, out_range=(0, 1))

# Option 2: Directly increase RGB values (scale them)
rgb_image = np.clip(rgb_image * 6.5, 0, 1)  # Increase brightness by scaling

# Option 3: Gamma correction specifically on the rgb_image
# gamma = 1.2 will make the image brighter; adjust as needed
gamma = 2.0
rgb_image = np.power(rgb_image, gamma)

# Plot segmented image with boundaries
plt.figure(figsize=(600 / 144, 600 / 144), dpi=144)
plt.imshow(mark_boundaries(rgb_image, segments))
plt.axis('off')  # Hide axes
# plt.show()
plt.savefig('./results/quick_2.png', bbox_inches='tight')
