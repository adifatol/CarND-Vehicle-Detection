import pickle
import numpy as np
import matplotlib
matplotlib.use('agg')
import pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC

# Getting back the calibration points:
with open('training_data/x_features.pickle','rb') as f:
    x_features = pickle.load(f)

is_nan = np.isnan(x_features.any())
is_inf = np.isfinite(x_features.all())

# Fit a per-column scaler - this will be necessary if combining different types of features (HOG + color_hist/bin_spatial)
X_scaler = StandardScaler().fit(x_features)
# Apply the scaler to X
scaled_X = X_scaler.transform(x_features)

#manually set after running feature extractor
len_car_features = 8792
len_noncar_features = 8968

# Define the labels vector
y = np.hstack((np.ones(len_car_features), np.zeros(len_noncar_features)))

# Randomize and Split data into training and test sets
sssplit = StratifiedShuffleSplit(n_splits=1, test_size=.2)
splited = sssplit.split(scaled_X, y)
for train_index, test_index in splited:
    X_train, X_test = scaled_X[train_index], scaled_X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# Use a linear SVC
svc = LinearSVC()
svc.fit(X_train, y_train)

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 3))

with open('training_data/trained_svm.pickle', 'wb') as f:
    pickle.dump([svc,X_scaler], f)

print('Trained model saved')
