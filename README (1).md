
# Lane Detection

Python implementation of lane detection using image processing algorithms in OpenCV.


### Code Requirements

* OpenCV (cv2)
* matplotlib
* numpy
### DataSet

The test image was taken from a road dataset video used in this [Kaggle project](https://www.kaggle.com/dpamgautam/video-file-for-lane-detection-project)
*
### License

[MIT](https://choosealicense.com/licenses/mit/)


### Image processing pipeline

1. Loading of test image
2. Detect edges (gray scaling, Gaussian smoothing, Canny edge detection)
3. Define region of interest
4. Apply Hough transforms
5. Post-processing of lane lines 
### Canny edge detection
The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images
The algorithm is based on grayscale pictures
The Canny edge detection algorithm is composed of 5 steps:

1. Noise reduction;
2. Gradient calculation;
3. Non-maximum suppression;
4. Double threshold;
5. Edge Tracking by Hysteresis.
### Noise Reduction

Edge detection results are highly sensitive to image noise.

To get rid of the noise on the image, Gaussian blur to smooth is applied to it. To do so, image convolution technique is applied with a Gaussian Kernel (3x3, 5x5, 7x7 etc…). 

The kernel size depends on the expected blurring effect. 
Basically, the smallest the kernel, the less visible is the blur. In our example, we will use a 5 by 5 Gaussian kernel.

Code to generate Guassian kernel: 

```python import numpy as np

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g
```


    
## Average and extrapolation of lines


Hough lines generated indicate multiple lines on the same lane. So, these lines are averaged to represent a single line. Also, some lines are partially detected. The averaged lines are extrapolated to cover the full length of the lanes.

The averaging process is performed based on the slopes of the multiple lines, which are to be grouped together belonging to either the left lane or the right lane. In the image, the y co-ordinate is reversed, (as the origin is at the top left corner) thus having a higher value when y is lower in the image. By this convention, the left lane has a negative slope and the right one has a positive slope. All lines having positive slope are grouped together and averaged to get the right lane, and vice versa for the negative slopes to obtain the left lane. Final lanes detected can be seen below:


![Hough_lines_avg](https://user-images.githubusercontent.com/79323560/193099506-d9728bb4-73e7-4bfe-aa58-3faec0d7449e.png)
