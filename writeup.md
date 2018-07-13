# **Finding Lane Lines on the Road** 


[image1]: ./examples/result.png "result"
[image2]: ./examples/bad_result.jpg "bad_result"
---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps. 
First, I converted the images to grayscale.
Second, I used canny edge detection algorithm to get the edge of the grayscale image.
Third, gaussian blur filter is been used to smooth the image to avoid noise.
Then, I extract the interest region from the image and get rid of unuseful region.
Then, I used hough transform to find the most probability lines of the image. 
Finally, I add the land line into the origin picture and get the final image.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function:
Instead of showing all probability lines, I have put these lines to three parts: right lines, left lines and other.
Then, the mean of slope and variance of right lines and left lines have been calculated.
Final, I separately used the mean of slope and variance of right lines and left lines to get the right line and left line.

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the picture quality is not very good.
For example, these images result is not good.

![alt text][image2]


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use the curve to replace the straight line. But maybe it is too diffcult.