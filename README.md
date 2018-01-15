# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[video1]: ./project_video.mp4
[sliding_windows]: ./figs/find_cars.png
[features]: ./figs/vehicle_features.png
[heatmaps]: ./figs/update_heatmap.png

---

### Feature Extraction

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The feature extraction routines and training script are located in VehicleDetectionFunctions.py and Training.ipynb respectively. The HOG features were extracted using the extract_features() and get_hog_features() functions from the class lesson placed in the VehicleDetectionFunctions.py file.

To test all the colorspaces, imported all the vehicle and non-vehicle training images and randomly selected images from both sets and plotted all the potential features (HOG, spatial binning, color histogram, etc...). Below is an example for the 'YCrCb' colorspace.

![alt text][features]

#### 2. Explain how you settled on your final choice of HOG parameters.

To test the efficacy of the features and their parameters, the I used the implemented SVC classifier's test accuracy as a benchmark and tweaked the parameters until I was able to achieve a sufficient test accuracy. After tweaking the parameters I settled on `orientations=9`, `pixels_per_cell=8`, `cells_per_block=8`, `spatial_size=(32,32)`, `n_bins=64` for the 'YCrCb' colorspace.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used a Linear SVC classifier for the simplicity and eifficiency. The classifier training was implemented in the Training.ipynb, which reads in the training data, generates all the chosen features, and formats and normalizes the data for the classifier. After training the classifier, I saved the SVC and Scalar function in a pickle file. 

With the YCrCb color space and using the spatial binning, histogram and HOB for all three channels, I was able to achieve a test result of 0.9899.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The implementation for the Sliding Window Search can be found in the VehicleDetector.py class implementation, which is responsible for finding cars with the sliding window implementation, calculating and updating the average heatmap, and also applying thresholding and labels to identify cars with a provided new consecutive input frame. 

Using the FindCars() routine provided in class, I added some modifications that allowed the sliding window to start at a specified x offset to mask out the left side of the image since the cars from incoming traffic were often detected in the distance. In a more complex system, it'd be possible to only slide the windows over our lanes, which could be identified using lane line detection. Or possibly add a feature class of cars to differentiate between oncoming and with-the-flow traffic. 

There was a large variation of scales used to cover the different depths where a car could exist in the image. For my implementation, I had scales of [1.0, 1.25, 1.5, 1.75, 2.0, 3.0]. Each sliding window had three vertical overlaps of 25% starting from the yoffset of 400px and slid from the specified xoffset(=512px) to the end of the image. The figure below shows the ROI for each scale and the corresponding cars found.

![alt_text][sliding_windows]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The above figure has examples of an accumlation of all the scaled windows that were found in the sliding windows. To optimize the classifier, I summed the calculated heatmaps and thresholded to filter out false positives.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

There were multiple strategies for filtering out false positives in my project and most of that implmentation exists in the VehicleDetector.py class. I restricted the sliding windows to the bottom right quadrant of the image to prevent cars on the other side of the highway from being detected. I also summed the last 10 frames with an exponential decay factor and used the averaged heatmap to threshold. I tested various thresholds and found a balance between inconsistent car detects and false positives. Because of the averaging, there were some cases where there were tiny boxes or slivers would branch off from a main heatmap blob so I created a minimum size of 50x50px for the drawing of the labels. 


### Here is a visualization of the process from the sliding windows to heatmap to labeled cars

![alt text][heatmaps]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The masking of the left side of the image was a contrived solution. Ideally the sliding windows should be restricted to a segment of the image that we know is considered our lane. We could also easily classify cars coming from the other direction and also measure it's velocity and direction to signify that it's a car in another lane. We could also make it more robust with more training data and by also creating more scaled windows for better resolution. 


