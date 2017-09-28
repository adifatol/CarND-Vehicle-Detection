##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png

[hog_c1]: ./output_images/train_hog_features/img1.png
[hog_c2]: ./output_images/train_hog_features/img2.png
[hog_c3]: ./output_images/train_hog_features/img3.png
[hog_c4]: ./output_images/train_hog_features/img4.png
[hog_c5]: ./output_images/train_hog_features/img5.png
[hog_c6]: ./output_images/train_hog_features/orig1.png
[hog_c7]: ./output_images/train_hog_features/orig2.png
[hog_c8]: ./output_images/train_hog_features/orig3.png
[hog_c9]: ./output_images/train_hog_features/orig4.png
[hog_c10]: ./output_images/train_hog_features/orig5.png

[hog_n1]: ./output_images/train_hog_features_car/img1.png
[hog_n2]: ./output_images/train_hog_features_car/img2.png
[hog_n3]: ./output_images/train_hog_features_car/img3.png
[hog_n4]: ./output_images/train_hog_features_car/img4.png
[hog_n5]: ./output_images/train_hog_features_car/img5.png
[hog_n6]: ./output_images/train_hog_features_car/orig1.png
[hog_n7]: ./output_images/train_hog_features_car/orig2.png
[hog_n8]: ./output_images/train_hog_features_car/orig3.png
[hog_n9]: ./output_images/train_hog_features_car/orig4.png
[hog_n10]: ./output_images/train_hog_features_car/orig5.png

[slide1]: ./output_images/image_slide_windows/img1_far.png
[slide2]: ./output_images/image_slide_windows/img1_med1.png
[slide3]: ./output_images/image_slide_windows/img1_near.png

[foud_cars1]: ./output_images/found_cars/img2.png
[foud_cars2]: ./output_images/found_cars/img3.png
[foud_cars3]: ./output_images/found_cars/img4.png

[heatmap1]: ./output_images/heatmap_cars/img4.png
[heatmap2]: ./output_images/heatmap_cars/img12.png

[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Project Structure

For this project I used python scripts. The implementation is modular so we can run various steps independently.

I first implemented a [feature extractor](https://github.com/adifatol/CarND-Vehicle-Detection/blob/master/feature_extractor.py) which will extract the features HOG features and save it into the training_data folder.

The second script is the [trainer](https://github.com/adifatol/CarND-Vehicle-Detection/blob/master/trainer.py). This is where a SVM model is trained using the extracted features and the model parameters are saved in the same training_data folder.

The second script is the [image pipeline](https://github.com/adifatol/CarND-Vehicle-Detection/blob/master/pipeline_img.py). This pipeline will use the trained SVM model and apply it on the test images.

The last script is the [video pipeline](https://github.com/adifatol/CarND-Vehicle-Detection/blob/master/pipeline_video.py). Same as the image pipeline, this will process the project video and generate the processed one.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the [feature extractor](https://github.com/adifatol/CarND-Vehicle-Detection/blob/master/feature_extractor.py) script. In the [modules](https://github.com/adifatol/CarND-Vehicle-Detection/tree/master/modules) folder there is a library [lesson functions](https://github.com/adifatol/CarND-Vehicle-Detection/blob/master/modules/lesson_functions.py) which contains the functions discussed in the lectures (`extract_features`, `get_hog_features` etc).

The features extractor starts by reading in all the `vehicle` and `non-vehicle` images. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][hog_c1] ![alt text][hog_c2] ![alt text][hog_c3] ![alt text][hog_c4] ![alt text][hog_c5]

![alt text][hog_c6] ![alt text][hog_c7] ![alt text][hog_c8] ![alt text][hog_c9] ![alt text][hog_c10]

![alt text][hog_n1] ![alt text][hog_n2] ![alt text][hog_n3] ![alt text][hog_n4] ![alt text][hog_n5]

![alt text][hog_n6] ![alt text][hog_n7] ![alt text][hog_n8] ![alt text][hog_n9] ![alt text][hog_n10]

#### 2. Explain how you settled on your final choice of HOG parameters.

I started with the parameters from the lesson. When I finished the implementation of the full image pipeline (used on the test images) I tried various combinations of orientations, pixels per cell, spatial_size and hist_bins. For some combinations the feature extraction was faster but did not give as good results during the classification as for others. I settled with the parameters found in [spatial_configs.py](https://github.com/adifatol/CarND-Vehicle-Detection/blob/master/modules/spatial_configs.py)

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the LinearSVC class in the [trainer.py](https://github.com/adifatol/CarND-Vehicle-Detection/blob/master/trainer.py).

Before the actual trainig, the features previously extracted were loaded. As the features contained a combination of HOG, color_hist and bin_spatial, I used `StandardScaler` for normalization. The data contains both car and noncar features ordered as they were downloaded so the array needed shuffling. Using `StratifiedShuffleSplit`, I was able to also split the data into traing and test sets as 0.8/0.2.

The trained model has an accuracy of 0.994 and after traing was saved for future uses.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

For sliding windows I used the slide_window function from the [lesson](https://github.com/adifatol/CarND-Vehicle-Detection/blob/master/modules/lesson_functions.py)

I decided to setup multiple runs with various window sizes: smaller for faraway vehicles and larger for closer ones. In the next images there are some examples of this method, but after testing different sizes and overlaps in the video, the final ones have slightly different positions and sizes.

![alt text][slide1]

![alt text][slide2]

![alt text][slide3]

I decided to select the sizes of the windows and the overlap based on the number of predictions and false positives.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][foud_cars1]
![alt text][foud_cars2]
![alt text][foud_cars3]
---

These results were improved when working on the video pipeline by updating the sliding windows sizes/overlaps.

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_processed_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap that I saved into a deque object. This object records the heatmaps from the last 10 frames.

I then averaged the heatmaps in order to filter the false positives.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video at various stages (cumulated `heat`):

![alt text][heatmap1]
![alt text][heatmap2]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The initial results were quite bad because the settings of the image overlaps and the sizes of the sliding windows. After I improved these, the predictions were better but the bounding boxes were "wobbly" and more false positives started to appear.
The solution for this was to use the averaging of the heatmap over the last ~10 frames. Also, using YCrCb colorspace seemed to have improved the results on white cars.

Probably the pipeline will fail on very close vehicles, especially when they are not fully present in the frame. Another problem seems to be the performance of the processing time, which could be improved for example by using different overlaps values based on window sizes (now all have the same value of 0.75).
