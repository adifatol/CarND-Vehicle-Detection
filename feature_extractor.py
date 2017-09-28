import glob
import pickle
from modules.lesson_functions import *
from modules.spatial_configs import *

cars = glob.glob('training_data/vehicles/**/*.png')
noncars = glob.glob('training_data/non-vehicles/**/*.png')

car_features = extract_features(cars, color_space=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_size=spatial_size, hist_bins=hist_bins)

notcar_features = extract_features(noncars, color_space=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_size=spatial_size, hist_bins=hist_bins)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

with open('training_data/x_features.pickle', 'wb') as f:
    pickle.dump(X, f)
