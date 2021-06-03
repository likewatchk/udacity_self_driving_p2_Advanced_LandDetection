## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/straight_lines1_undist.jpg "Undistorted"
[image2]: ./output_images/straight_lines1_result.jpg "result_1"
[image3]: ./output_images/straight_lines2_result.jpg "result_2"
[image4]: ./output_images/test1_result "result3"
[image5]: ./output_images/test2_result "result4"
[image6]: ./output_images/test3_result "result5"
[video1]: ./project_video.mp4 "Video"


### Overall
`./P2.ipynb` is my project code.
there are each codes according to the steps for 'finding lane'
And, There are pipelines for single image and video.

#### You can see all of the images that correspond to all the intermediate results in `P2.ipynb`
#### please see the intermediate result's images in `P2.ipynb` .


### Camera Calibration

Camera Calibration step needs some `Chessboard images`. In this project, there are the images you gave for camera calibration. I used these images. 
</br></br>
1. corners finding loop.</br>
  a. read image.  `mpimg.imread` or `cv2.imread` </br>
  b. change it to gray-scale. `cv2.cvtColor(RGB2GRAY)` or `cv2.cvtColor(BGR2GRAY)`</br>
  c. find corners. `cv2.findChessboardCorners`</br>
  d. collect corners and object points.</br>
2. calibrate camera using collected corner-points and object points</br>
  (object points is the points with real-world coordinate pairing with the corner-points(image-points).</br>
  ```python
  # after finding & collecting corner-points(image-points).
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points_list, image_points_list, img_size, None, None)
  ```
  
  (You can see the chessboard images with corners drawn in `P2.ipynb`)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

In calibration step, I found parameters for camera calibration.</br>
I used these params so that I can get the `undistorted image`. This is the example.</br>
![alt text][image1]

#### 2. color transforms, gradients or other methods to create a thresholded binary image.

I used a combination of color and gradient thresholds to generate a binary image.</br>
It took a lot of time to find these proper thresholds, but these threshold are only useful to image and project_video, not to challenge_video and harder_challenge_video. So, Originally, I have to tune it more, but I want to move on to the next course because I spended almost 2 monthes for this course... OMG</br>
(You can see binary images in `P2.ipynb`)


#### 3. perspective transform

I found the proper source points and destination points for perspective transform for bird-eye-view.</br>
I made some parameters for it, which are named "blank_side_upper", "blank_side_below", "height". I think It is necessary for the trapezoidal shape used in this step(perspective warping) to be symmetrical. So, the params i made is designed for it.</br></br>

I found some points of trapezoidal shape for warp.</br>
It took a lot of time to find these proper points, but these points are only useful to image and project_video, not to challenge_video and harder_challenge_video. So, Originally, I have to tune it more, but I want to move on to the next course because I spended almost 2 monthes for this course... OMG</br></br>

```python
src = np.float32([[blank_side_below+xdir_move,img_size[1]], [blank_side_upper+xdir_move, clip_height], 
                  [img_size[0]-blank_side_upper+xdir_move, clip_height], [img_size[0]-blank_side_below+xdir_move, img_size[1]]])
dst = np.float32([[blank_side_below+xdir_move,img_size[1]], [blank_side_below+xdir_move, 0], 
                  [img_size[0]-blank_side_below+xdir_move, 0], [img_size[0]-blank_side_below+xdir_move, img_size[1]]])

```
I got the M, Minv for warp from `cv2.getPerspectiveTransform` and I got the warped image by `cv2.warpPerspective`</br>
(You can see the transformed binary in `P2.ipynb`)</br>

#### 4. identifying lane-line pixels and fit their positions with a polynomial?
1. find the first bases(left_base, right_base) point for starting to finding lane pixels by historgram of bottom-half of image.</br>
2. extract nonzero indices of the image. `nparray.nonzero()`</br>
3. find the lane pixels by window. This step is implemented with loop. The number of loops is `nwindows` and `margin`is window's width and `window_height` is calculated by dividing image_size(y direction, image height) by 'nwindows'. So, by this step, I could find the indices of nonzero fit to the criteria(in the window). additionally, If I found more pixels in a window than the `min_pix`, I had to find new x_bases for windows.</br>
4. After finding lane pixels, I could fit these points. `np.polyfit`</br>

(You can see the transformed binary in `P2.ipynb`)</br>

#### 5. calculate the radius of curvature of the lane

I can use the formula for radius of a point on polynomial.
```python
lx = leftx * xm_per_pix
ly = lefty * ym_per_pix
rx = rightx * xm_per_pix
ry = righty * ym_per_pix
left_fit_cr = np.polyfit(ly, lx, 2)
right_fit_cr = np.polyfit(ry, rx, 2)

lA = left_fit_cr[0]; lB = left_fit_cr[1]; lC = left_fit_cr[2]
rA = right_fit_cr[0]; rB = right_fit_cr[1]; rC = right_fit_cr[2]
left_curve_radius = ((1+(2*lA*y_val + lB)**2))**(3/2) / np.abs(2*lA)
right_curve_radius = ((1+(2*rA*y_val + rB)**2))**(3/2) / np.abs(2*rA)
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

---

### Pipeline (video)

Video has a sequence of pictures. The polynomial of lane in some picture is familiar with the previous polynomial. So, I used 'previous fit'.

result1, 2, 3 was made by the algorithm with previous fit.</br>
result2, 3 was made by the algorithm without previous fit. It took more time to get result.</br>
but, using previous fit has some risk because if algorithm find wrong result in some picture, it could effect the result of the next picture.</br>

Here's a [project_video result](./output_videos/project_video.mp4)</br>
Here's a [challenge_video result](./output_videos/challenge_video.mp4)</br>
Here's a [harder_challenge_video result](./output_videos/harder_challenge_video.mp4)</br>
Here's a [challenge_video result2](./output_videos/challenge_video2.mp4)</br>
Here's a [harder_challenge_video result2](./output_videos/harder_challenge_video2.mp4)</br>

---

### Discussion

#### difficulties
1. finding the proper threshold for binary having almost all of the lane points was hard. the threshold i found was useful to `project_video` but not to `challenge_video`.
2. finding the proper trapezoid for perspective warp was hard. The trapezoid I set was useful to `project_video` but not to `harder_challenge_video`
3. If I have more time, I want to find the more proper threshold and trapezoid but I spended almost 2 monthes in this course... I want to move on to the next course, "deep learning"
4. There are many hurdles for finding lane lines stably. especially, brightness is very powerful condition. so, if there are some shade by trees or buildings or bridges, It effects the quality of result. And, sometimes lane is very unclear. It also effected to the results.
5. To use the 'lane-detection' actually, It should excute it's job in 'real-time'. but my algorithm take a lot of time to excute.

I realized 'Mobileye' has very powerful tech.
