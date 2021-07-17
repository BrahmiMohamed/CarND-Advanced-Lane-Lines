## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](Output.jpg)

In this project, the goal is to write a software pipeline to identify the lane boundaries in a video.

The steps of the implemented pipeline are:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images. using the dedicated function `Cam_Calib( )` whichs output `mtx` and `dist`.
* Apply a distortion correction to raw images.
* Use color transforms and gradients to create a thresholded binary image with help of the function `grad_thresh` in `lib_functions.py`.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

# Keeping track of left and right lanes 

To keep track of left and right lanes and store relevant attributes two instances of the class `Line()`are created and used. The main attributes of this calls are:
* detected: boolean variable to mark whether lane has been found or not  
* recent_xfitted = x values of the last valid n fits of the line
* current_xfitted = x values of the current n fits of the line
* bestx = average x values of the fitted line over the last n iterations 
* best_fit = polynomial coefficients averaged over the last n iterations  
* self.current_fit = values of the last n polynomian fits of the line

# How to filter out outlier and bad lanes

In order to achieve a smooth detection of the lanes, sanity check of the current detected line is used. Therefore the function `sanity_check` performs following checks:
* is the lane widht plasubile (in range [3m...4,7m])
* is the std of the lane widh plausible (< 0.5m), indicating parallel lines

Only in case both conditions are met, the sanity checks return `true` and the found line is used. Otherwise the last valid line is used.

# Calculating Curvature and Offset

Once valid lines are found, the x-fitted values of the right and left lines are used to calculate the curvature (using `measure_curvature_real( )`) and the offset to the center.
These values are then added to the final image which is in turn written into the output video.


