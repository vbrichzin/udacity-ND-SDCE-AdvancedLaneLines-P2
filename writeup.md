# Advanced Lane Finding Project

The goals / steps of this project were the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration2.jpg "Distorted"
[image2]: ./camera_cal/undistort_calibration2.jpg "Undistorted"
[image3]: ./test_images/test6.jpg "Road Distorted"
[image4]: ./output_images/output_undistorted_test5.jpg "Road Undistorted"
[image5]: ./output_images/output_binary_test5.jpg "Binary"
[image6]: ./output_images/output_orig_src_straight_lines1.jpg "Undistorted SRC"
[image7]: ./output_images/output_warped_dst_straight_lines1.jpg "Warped DST"
[image8]: ./output_images/output_binary_polynomial_test5.jpg "Binary Polynomial"
[image9]: ./output_images/output_binary_polynomial_with_cr_test5.jpg "Binary Polynomial with Curvature and Offset"
[image10]: ./output_images/output_result_test5.jpg "Result"

[video1]: ./output_test_videos/output_project_video.mp4 "Video"
[video2]: ./output_test_videos/output_project_filter_video.mp4 "Video with filter"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### I will also consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You are reading the write-up in the file `writeup.md`.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the Jupyter notebook located in `./P2.ipynb` (directly after the general imports).  

I also started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. I am limiting myself to a 9x6 chess board as was suggested in the project hints.
I am also assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection when I iterated through the calibration files.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to a test image using the `cv2.undistort()` function.

Here is the distorted chessboard image:

![alt text][image1]

And here is the undistorted chessboard image:

![alt_test][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I used the calibration parameters calculated in the cell before and calculated in cell #3 undistorted images of the original test images. Since the `cv2.undistort()`function was giving back an image in BGR I also needed to swap the color channels with `cv2.split()`and `cv2.merge()`.

Here is an original, distorted test image:

![alt text][image3]

And here is the undistorted version of it:

![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used the same combination of color and gradient thresholds to generate a binary image. I was able to re-use some of the code from the combined binary quiz in one of the lessons. In cell #4 of `P2.ipynb` the `binary_img()` function is using thresholds for the S-channel in the HLS color space and the the L-channel for the x-gradient and a threshold for the B-channel in the HSV color channel. All with the idea that the S-channel of HLS should help me identify the white lines, while the B-channel of HSV should help me identify the yellow lines. But before I do the color space transforms I use contrast correction to fight excessive darkness or brightness through the use of the `cv2.equalizeHist()` in the YUV color space (as was suggested by my reviewer).
With the combination of these thresholds into one binary image the result was showing the lane lines quite well. I then only removed some noise through the use of the morphological transfer function `cv2.equalizeHist()`. Even though other object edges on cars and trees were visible as well, with the later windowing approach the lane lines can be well separated from the rest.

Here is a binary image (same image as for the undistorting in the rubric point before) thus created:

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in cell #6 in `P2.ipynb`.  
I hardcoded the `src` and `dst` points outside the `warp()` function and calculated the `M` and `Minv` perspective transformation matrice for later usage. The `warp()` function takes as inputs an image (`img`), as well as the undistortion coefficients, and returns a perspectively transformed image.  I chose the hardcode the source and destination points in the same manner as suggested in the writeup_template file. How this would lead to a roughly 30m long and 3.7m wide corridor in the transformed image made sense to me (also trying to plausibilize the length of 30m with the number of dashed lines and intermittences).

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Here is the original, undistorted image with the polygon defined by the `src` points drawn in:

![alt text][image6]

And here is the warped image with the polygon defined by the `dst` points drawn in:

![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In cell #8 of `P2.ipynb` I defined two functions, the `find_lane_pixels()` and `fit_polynomial()`. When going through the list of test images in cell #9, I first undistort each image, the create a binary image, and since the `warp()` function is expecting a "color" image I am creating a binary color image, but then I take only the first channel of the returned image from the `warp()` function to feed into the `fit_polynomial()` function. This function is then calling the `find_lane_pixels()` function. Both functions, `find_lane_pixels()` and `fit_polynomial()`, I am re-using from previous quizzes, and what they do is that first the left and right lane positions are identified based on the maximums of the histogram of the lower half of the binary image fed into `find_lane_pixels()`, and then looking for non-zero values in the nine sliding windows. These identified non-zero pixel positions are then used by the `fit_polynomial()` function to fit a polynomial into the left and right identified lane lines.
Additionally the identified lane line pixels are colored in a color version of the binary picture (left line colored red, right line colored blue), the sliding windows is colored in and the fitted polynomials are drawn in in green.

Here is such a created image based on the `test5.jpg` test image:

![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines cells #10 and #11 in `P2.ipynb`. I used the values of 30m and 3.7m for the forward looking lenght and line width as was used in the previous quiz. The number of pixels for each I estimated based on the hardcoded perspective transformation points, `dst`.
For the curvature calculation I created a function `measure_curvature_real()` where I was able to re-use some of the code from the previous quiz. For the distance of the vehicle to the center of the lane I created a function `measure_vehicle_distance2center()`. To calculate this value I calculated the pixel in the middle of the lane and subtracted from that the pixel from the middle of the image as it was stated that the camera was mounted in the center of the vehicle.

I plotted these two values with `cv2.putText()` into the the binary image with the lane lines identified and the fitted polynomial and sliding search windows:

![alt text][image9]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell #12 in `P2.ipynb`.  I didn't create a function, but implemented straight into the loop for the images with the code snippet taken and adjusted from the project hints. This included first to draw a polygon between the fitted left lane polynomial and the fitted right lane polynomial into a new, still "warped" image. Then, in a second step, I used the previously calculated inverse perspective transformation matrix `Minv` to "warp" back this new image to the original camera perspective. This image was then overlaid with the original, undistorted image. Plus the calculated radius of curvature and the offset of the vehicle to the center of the lane was printed into the image using `cv2.putText()` function. 

Here is an example of the final result of my pipeline executed on the test image `test5.jpg`:

![alt text][image10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Contrary to rubric point #6 I then put the entire pipeline into one function `process_image()` that calls the individual functions. The `process_image()` function is then called for every frame.

Here's a [link to my video result][video1]

I also made use of a limited version of the class definition for a lane line to be able to filter to a certain degree. At least in the project video on the highway the change of curvature is pretty small, thus I let newly calculated values only contribute to a certain degree to the fitted lane. This helped to avoid jumpy changes and made it more steady. It worked well on the project video, and also the challenge video improved a little bit. Yet on the harder challenge video this approach was failing as there are bigger changes in curvature on the windy road.

Here is a [link to my improved project video result with use of a filter][video2]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I would say that my pipeline works reasonably well for the normal `project video`.
In the `challende video` and even more the `harder challenge video` it shows that it still falls short for more complicated situations with different color of pavement and especially difficult lighting conditions and steeper radii of curvature where the sliding windows are all on the edge and trigger false detections.

Also, I did not implement the `Line()` class that would keep track of the lane lines and make the searches faster if not the sliding windows are used, but the region search around the currently identified lane lines. This should definitely speed up the search which is right now clearly not real-time capable and takes quite long. Also, by use of filtering I could make the detection more robust and less jumpy.

In the end I think I would need to start and think about a good measure to estimate the "quality" of the detection, maybe by the quality of the polynomial fit and the sanity checks and then use that measure to weight the next detection more or less heavily into the average. This approach should filter the result strongly and provide for a much less jumpy detection (also of the radius of curvature and offset position that shouldn't jump as much as they do from frame to frame).
