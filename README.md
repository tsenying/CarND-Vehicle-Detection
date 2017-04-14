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
[image3]: ./output_images/sliding_window_scale_1.0.png
[image4]: ./output_images/sliding_window_scale_1.5.png
[image55]: ./output_images/sliding_windows.png
[image66]: ./output_images/heatmap_threshold.png


---

## Project Files

- README.md : this file
- src/* : Python source files
- svc_trained.p : Trained SVM Linear Classifier pickle file
- [Vehicle detection result video](https://youtu.be/uWMWS8afEXw)

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
The general approach for dealing with car varying sizes in an image is to use the **sliding window** approach.
For a given *scale*, a sub-image is cut out from the whole image and examined to see if it contains a vehicle.
A given scale *window* is slid across the image with some overlap until the entire image is examined.  

Features relevant to distinguishing a car from the background is extracted from the sub-image,
the sub-image is determined to be a *car* or *not-car* by a classifier.

The classifier is pre-trained with images known to be cars or non-cars.

Vehicle detection processing steps in order are:
1. Image **feature extraction**
2. Feature **Classification** of image features
3. **Detection** at different positions and scales  
	Vehicles appear at different scales depending on distance, and different positions on road surface.
4. Vehicle position **History Influence**  
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

Spatial binning is a technique for capturing the spatial data of an image while reducing the number of features.
It is done simply by reducing the size of an image while still retaining spatial signature.
In our case, the original 64x64 is reduced on both dimensions by half, reducing the number of features to 1/4 of original size.  
The spatial binning code is in function `bin_spatial` in [feature_extraction_utils.py](./src/feature_extraction_utils.py):line 34

### Color Histogram
As the name implies, color histogram is another binning technique that captures the color signature of an image,  
while reducing the number of features into the number of bins in the histogram.  
The color histogram code is in function `color_hist` in [feature_extraction_utils.py](./src/feature_extraction_utils.py):line 42

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
The general approach for dealing with car varying sizes in an image is to use the *sliding window* approach.
For a given *scale*, a sub-image is cut out from the whole image and examined to see if it contains a vehicle.
A given scale *window* is slid across the image with some overlap until the entire image is examined.

### Search Area
- The upper part of the image above the horizon does not need to be searched.
- Search areas vary by scale, smaller scales are more distant and occupy a narrow horizontal band near the horizon.

The horizon for the test video is at about y=400, so the search window y value is set to start here,  
the end y value varies with scale, with smaller scales requiring narrower bands to be searched.

#### Scales Used
A range of scales was tested with the following results:
- 0.25 produces many spurious false positives, requires long compute times.
- 0.5 produces mostly reasonable positives with some false positives
- 0.75 comparable to 0.5, detects white car better, but more false positives
- 1.0 does about as well as 0.75 with less false positives, distant cars are detected.  
  Search area y end value can be reduced to 528 at this scale. ![alt text][image3]  
- 1.5 distant cars start to drop off, e.g. in test image 3, the white car is not detected.  
  Search area y end value is set to 560 at this scale. ![alt text][image4]
- 2.0 good detection of mid-range cars with little false positives.  
  Search area y value is basically set to entire image below horizon at this scale.
- 3.0 may pick up near distance cars

The scales selected are [1.0, 1.5, 2.0]  
Smaller scales produce many false positives and require more compute time.  
Larger scales may pick up more near distance cars.

For the scales used, starting at 1.0, the true positive detections can extend the entire width of the image,
so the x start and stop values are set for the entire width of the image.

#### Window Overlap
0.75 overlap was chosen as a tradeoff between detecting cars well enough and incurring more compute resources if more overlap was used.

**Code**  
The `scale` variant was tested using [hog_subsample.py](src/hog_subsample.py),
the `find_cars` function in this file, which implements HOG subsampling to reduce computation, is also used in the video processing pipeline.

The scales and overlap used and search window per scale is visualized here,  
scales color: blue = 1.0, green = 1.5, red = 2.0
![alt text][image55]


#### Features Used

From the investigations described, the features that produced good results and used for the video processing pipeline are:  
- YCrCb 3-channel HOG features at 3 scales,  
- plus spatially binned features 
- and histograms of color in the feature vector.

### False Positive Filtering using Heatmap Thresholding

The classifier computes *false positives* that need to be filtered out.  
Most *false positives* are transitory and do not persist much beyond a single frame.

The **heat-map thresholding** approach can be used to filter out *false positives*.  
- Bounding boxes for detections are overlayed on a single blank image. [heatmap_threshold_detection.py#add_heat:line 11](src/heatmap_threshold_detection.py)  
- The resulting heat-map is thresholded to reject pixels below the threshold. [heatmap_threshold_detection.py#apply_threshold 21](src/heatmap_threshold_detection.py)  
- Bounding boxes are then calculated for remaining non-zero pixels, these boxes should represent detected cars.  
Done using [label](https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html) function from the [SciPy](https://docs.scipy.org) library.

This approach is visualized here for the test images:
![alt text][image66]
---

## Video Processing Implementation

####1. [Vehicle detection result video](https://youtu.be/uWMWS8afEXw)

####2. Vehicle position smoothing using history
Classifier results from frame to frame vary significantly leading to significant variation in bounding boxes for vehicles.
Past history of vehicle positions can be used to smooth out the vehicle bounding boxes of the video frame sequence.  

The approach used is to keep the history of classifier positive detection bounding boxes for a number of frames during video processing,  
for each frame, the current set of classifier positive detection bounding boxes is combined with past history to compute 
the current set of vehicle bounding boxes. [vehicle_detection.py 54](src/vehicle_detection.py)  


---

###Discussion

####1. Problems and Issues

As with any classification problem, choosing the right set of features for classifier training and prediction is key.
We chose to use the **YCrCb** _color space_ for HOG features although some other color space may work as well or in combination.  

Color **spatial binning** features and **Color Histogram** features were also used.
There is a tradeoff between extracting enough features to distinguish between cars and non-cars and too many features that result in high computation cost.  

#### Where will your pipeline likely fail?  
1. A sufficient training set is vital in providing correct classification.  
The small training set produced a trained classifier that was unable to detect white cars.  
Even the large training set appear skewed towards dark color cars and performed poorly on detecting white cars.

2. Structured images such as railings and lane lines produced false positives.

#### What could you do to make it more robust?

The LinearSVC SVM classifier with default settings was used.
Other settings and/or classifiers could be explored.

#### Vehicle tracking/occlusion/separation
Individual vehicle positions should be tracked and predicted.  
This would allow vehicle image merging and occlusions to be handled.
[Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter) may be a good approach.

#### Investigations
- channels used for features
- scale windows
- SVM decision function
- history averaging
- false positives
137,1047
