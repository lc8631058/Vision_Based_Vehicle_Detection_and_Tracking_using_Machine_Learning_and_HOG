##

---

**Vision based Vehicle Detection and Tracking using Machine Learning and HOG**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector, to compare which kind of combination could lead to the best result
* Note: for those first two steps don't forget to normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/test_imgs.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./examples/gradient_features.jpg
[image9]: ./examples/gradient_histo.jpg
[image10]: ./examples/hog-visualization.jpg
[image11]: ./examples/Unknown.png
[image12]: ./examples/Unknown-2.png
[image13]: ./examples/Unknown-3.png
[image14]: ./examples/Unknown-4.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in my codes) I extracted HOG features from the training images.

The code for this step is contained in the function `get_hog_features` of the extract_features.py  

I started by reading in all the `vehicle` and `non-vehicle` images.  This images are from [GTI vehicle image database](http://www.gti.ssr.upm.e`s/data/Vehicle_database.html) and [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/). Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Using gradient you can get the directions of pixels, such like this:

![alt text][image8]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). `orientations` means how many orientations you want to specify when apply HOG detecting, and then the algorithm will count the gradient magnitude of each orientations:

![alt text][image9]

And you can see what does parameter `pixels_per_cell` and `cells_per_block` do when apply HOG searching:

![alt text][image10]
 
I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. How I settled on my final choice of HOG parameters.

I tried various combinations of parameters, at first I combined all three features together to one vector, use the `StandardScaler()` to scale all features with different value ranges to a unified scale. Then experiment with different parameters such like:

```python
colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL'
spatial_size=(16,16)
hist_bins=32
hist_range=(0,256)
```

I have tried with a lot of different combinations of parameters, I found that `colorspace` values a lot for vehicle detection, so you should choos a really appropriate colorspace; The `pix_per_cell`, `spatial_size` and `hist_bins` normally influence the processing speed, but have lower influence on the final detection results. So the most important thing is your classifier and the HOG features, after many many experiments and references, I decide only to use HOG features to train my classifier, because it's stable and much faster. So my final parameters are:

```python
colorspace = 'YUV'
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL'
```

#### 3. Describe how (and identify where in my codes) to train a classifier using the selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features, cause there is only one kind of feature, you don't need to add a scaler first. At first I split the data into training (80%) and test sets (20%). Then I use `grid_search.GridSearchCV` to search for the best parameter combination of SVM, I tried with following combinations:

|      ('rbf', C=1)    		|     	  ('rbf', C=10)      					| 
|('linear', C=1)|('linear', C=10)| 

And `('linear', C=1)` has the best results, so I choose this to train a linear SVM with probability. You can refer that from the cells with title "Train the SVM".

### Sliding Window Search

#### 1. Describe how to implement a sliding window search.  How did I decide what scales to search and how much to overlap windows?

The `find_cars` function in extract_features.py describe how I implement sliding window to search for cars. First you should determine the ROI (Region of Interest), winthin ROI, implement the slide window search with window size 64x64 from left to right, top to down, block by block to search for cars. 

A fixed size of window won't lead to a comprehensive searching, cause the cars which are far away from you are smaller than the cars near by you, so to set a `scale` parameter to control the searching window size, if scale bigger than 1, the searching window will become bigger, smaller than one, window becomes smaller, such like what showed in following image:

![alt text][image3]

So after experiments, the final determined searching area are, for more details please refer to `show_detect_area` function:

![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 scales, each scale includes 2 kind of searching areas, and using YUV 3-channel HOG features in the feature vector, which provided a nice result.  Here are some test results on test images:

![alt text][image4]
---

### Video Implementation

#### 1. The final output videos:

Here's a [link to my video result](./test_videos_output/output_project_video.mp4)
There are no detected false-positives and all cars are be deteced and tracked all the time.


#### 2. Describe how (and identify where in my code) I implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

The detection and tracking will be much stable if you use every 15 frames' results to detemine the car position at every frame, here I use the `collections.deque(maxlen=15)` to deal with that mind: you save the heatmap of every 15 frames, you sum them up, you set a threshold to filter the false-positive area, because f-p won't be deteced in every frame (otherwise your classifier is really bad, man), so after thresholding the only in single frames detected f-p will be filtered; The other way to avoid f-p area is to set a probability-threshold for SVM, that means only the by SVM deteced areas which have the probability-scores more than fixed number will be judged as cars finally. 

---

### Discussion

#### 1. Briefly discuss any problems / issues I faced in my implementation of this project.  Where will my pipeline likely fail?  What could I do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

At first I use HOG to extract features from cars- and non-cars dataset, use the extracted features to train a SVM, then use slide window to search for small blocks from video frames, at the same time use the trained SVM to make classification: whether this block belongs to car or non-car. At the beginning I tried a lot of combinations of features, but all failed, no matter which kind of combinations I choose, I can't get the good results without false-positives, that's really trick, and finally I determined that my classifier has some problems, and it's because the extracted features can't reflect the differences between cars and non-cars, and the problem is, for HOG searching, I used L2 as the block normalization, then I use L1 as the block normalization, I get a good performed SVM, and good results. For the future I plan to try with different vedios to see if my pipeline can work on different environments.

