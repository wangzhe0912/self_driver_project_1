# -*- coding: UTF-8 -*-
"""
# WANGZHE12
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    left_list = []
    right_list = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y1 - y2) / (x1 - x2)
            if 0.5 < k < 0.7:
                # right
                right_list.append([x1, y1, x2, y2])
            elif -0.7 < k < -0.5:
                # left
                left_list.append([x1, y1, x2, y2])
            else:
                # stash data
                pass
    # separate calculate the sum of k, b of left line and right line
    try:
        left_k_sum = 0
        left_b_sum = 0
        left_min_point = (1000, 0)
        left_max_point = (0, 0)
        for line in left_list:
            x1, y1, x2, y2 = line
            k = (y1 - y2) / (x1 - x2)
            b = y2 - k * x2
            left_k_sum += k
            left_b_sum += b
            if x1 < left_min_point[0]:
                left_min_point = (x1, y1)
            if x2 > left_max_point[0]:
                left_max_point = (x2, y2)
        left_k_average = left_k_sum / len(left_list)
        left_b_average = left_b_sum / len(left_list)
        left_min_x = int((img.shape[0] - left_b_average) / left_k_average)
        cv2.line(img, (left_min_x, img.shape[0]), (left_max_point[0], left_max_point[1]), color, thickness)
    except Exception as e:
        print(str(e))

    try:
        # separate record the point of left line and right line
        right_k_sum = 0
        right_b_sum = 0
        right_min_point = (1000, 0)
        right_max_point = (0, 0)
        for line in right_list:
            x1, y1, x2, y2 = line
            k = (y1 - y2) / (x1 - x2)
            b = y2 - k * x2
            right_k_sum += k
            right_b_sum += b
            if x1 < right_min_point[0]:
                right_min_point = (x1, y1)
            if x2 > right_max_point[0]:
                right_max_point = (x2, y2)
        right_k_average = right_k_sum / len(right_list)
        right_b_average = right_b_sum / len(right_list)
        # left_min_point and right_max_point can be calculate by k, b and image shape
        right_max_x = int((img.shape[0] - right_b_average) / right_k_average)
        cv2.line(img, (right_min_point[0], right_min_point[1]), (right_max_x, img.shape[0]), color, thickness)
    # print("Left line: " + str((left_min_x, img.shape[0])) + " " + str(left_max_point) + " " + str(left_k_average) + " " + str(left_b_average))
    # print("Right line: " + str(right_min_point) + " " + str((right_max_x, img.shape[0])) + " " + str(right_k_average) + " " + str(right_b_average))
    except Exception as e:
        print(str(e))


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    # Step1: gray_image
    gray_image = grayscale(image)

    # Step2: canny
    canny_image = canny(gray_image, 50, 150)

    # Step3: gaussian blur
    gaussian_blur_image = gaussian_blur(canny_image, 5)

    # Step4: region_of_interest
    imshape = gaussian_blur_image.shape
    vertices = np.array([[(0, imshape[0]), (500, 300), (501, 300), (imshape[1], imshape[0])]], dtype=np.int32)
    interest_region_image = region_of_interest(gaussian_blur_image, vertices)

    # Step5: hough_lines
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = 2 * np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 80  # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40  # minimum number of pixels making up a line
    max_line_gap = 3  # maximum gap in pixels between connectable line segments
    # line_image = np.copy(image)*0
    lanes_image = hough_lines(interest_region_image, rho, theta, threshold, min_line_len, max_line_gap)

    # Step6: weighted_img
    result = weighted_img(lanes_image, image, α=0.8, β=1., γ=0.)
    return result


if __name__ == "__main__":
    image = mpimg.imread('test_images/solidWhiteRight.jpg')
    result = process_image(image)
    plt.show(result)
    # plt.savefig("./examples/resultSolidWhiteRight.jpg")
