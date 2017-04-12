# Udacity CarND Vehicle Detection Project

The goals of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* In addition, apply a color transform to each video frame image and then append binned color features, as well as histograms of color, to the HOG feature vector. 
* For the first two steps, normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run pipeline processing on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/hog_RGB_12_8_2.png
[image2]: ./output_images/hog_YCrCb_12_16_2.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---

## Project Files

- README.md : this file
- src/* : Python source files
- svc_trained.p : Trained SVM Linear Classifier pickle file
- https://youtu.be/uWMWS8afEXw : vehicle detection result video

## Usage:

Create sklearn.preprocessing.StandardScaler from train data
and train sklearn.svm.LinearSVC linear SVM classifier from train data,  
saved results to svc_trained.p
```
python src/search_and_classify.py
```

Run video processing pipeline, and output results to 'cars_video.mp4':
```
python src/process_video.py 
```
## Video Vehicle Detection Overview
Vehicle detection in a stream of images from a camera or video has these aspects:

Cars in a image can be at different distances, the same car will have varying apparent sizes inversely proportional to distances.
The general approach for dealing with car varying sizes in an image is to use the *sliding window* approach.
For a given *scale*, a sub-image is cut out from the whole image and examined to see if it contains a vehicle.
A given scale *window* is slid across the image with some overlap until the entire image is examined.  

Features relevant to distinguishing a car from the background is extracted from the sub-image,
the sub-image is determined to be a *car* or *not-car* by a classifier.

The classifier is pre-trained with images known to be cars or non-cars.

Vehicle detection processing steps in order are:
1. Image Feature Extraction  
2. Feature Classification  
	What classifier to use
3. Scaling and Sliding Window  
	Vehicles appear at different scales depending on distance from camera
4. Frame History Influence  
	Past vehicle location indicates future location.  
	
##1. Feature Extraction: Histogram of Oriented Gradients (HOG), Spatial Binning, Color Histogram 
*What features to extract?*
The features extracted from a sliding window sub-image are determined to be useful in distinguishing cars from non-cars.  
The features used include: *Histogram of Oriented Gradients (HOG)*, *Spatial Binning*, and *Color Histogram*. 
	
### Histogram of Oriented Gradients

The [HOG feature descriptor](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html) processing
divides the image into sub-cells. For each cell, the gradient is detected, then a histogram of gradients is computed.  
This results in a set of histograms for the image.

These histograms provide a signature that differentiates between cars and non-cars.

Code for HOG feature detection is in function `get_hog_features` [feature_extraction_utils.py](./src/feature_extraction_utils.py):line 14  

Here is an example of HOG features for the RGB color space for one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Here the color space is `RGB` and HOG parameters used are `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`

Another permutation of parameters using color space 'YCrCb' is shown here:

![alt text][image2]

The above example shows a larger pixels_per_cell size leads to lower detail captured.

Different color spaces were investigated, including RGB, HSV, YUV, YCrCb as input to the HOG function.  
Varying HOG parameters were experimented.

#### Choice of HOG parameters
HOG parameters include:
- orient : number of directions around the clock to use.  
	9 seems to be deemed sufficient, but using 12 avoids direction bias in both left/right and top/down directions.
- pix_per_cell : pixels in a rectangular area to collect direction and magnitude.  
    a larger number, such as 16, means fewer cells per image and seems to lead to loss of detail,  
    8x8 cells appear to provide enough detail
- cell_per_block : this parameter is used to normalize across neighboring cells and filters noise from illumination, shadowing, and edge contrast.  
    This parameter has minimal effect on most images. Instead of using a large number of cells and incurring computation cost, 2 is a good setting to start with.

### Spatial Binning

### Color Histogram

## Classifier
Classifiers are used to process an input feature set and determine output class.
A number of classifiers are provided by the scikit-learn machine learning library, including:
1. [Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html) - can be brittle across feature spaces
2. SVM - a algorithm that maximizes hyper-plane separation between classes.  
	A variant that scales well to large number of samples is [LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
3. [Decision Tree](http://scikit-learn.org/stable/modules/tree.html) - tends to produce uneven decision surfaces.

We start with the LinearSVC classifier as it seems the most robust.

####3. Classifier Training
A set of cars and not-cars images were used to train the classifier.

This is done with the python script [search_and_classify.py](src/search_and_classify.py)

Training steps involved are:
- Extract features for cars and non-cars, including HOG, spatial binning and color histograms, from each image.  
  Done with function `extract_features` in [feature_extraction_utils.py](src/feature_extraction_utils.py):line 54  
  
- Normalize feature data using [sklearn StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)  
  First fit the scaler [search_and_classify.py](src/search_and_classify.py):line-164  
  Then normalize the feature data [search_and_classify.py](src/search_and_classify.py):line-166

- Split sample data into train and test sets [search_and_classify.py](src/search_and_classify.py):line-174

- Train the classifier [search_and_classify.py](src/search_and_classify.py):line-174

The resulting scaler and classifier are saved into a pickle file for use in the video processing pipeline.
  
## Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Processing Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

