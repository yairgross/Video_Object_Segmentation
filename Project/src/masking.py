import cv2
import numpy as np


def new_frame():
	cap = cv2.VideoCapture('input_video.mp4')
	ret, frame = cap.read()
	cv2.imwrite('first_frame.jpg', frame)


def find_colors():
	# find hsv color boundaries
	bgr = np.uint8([[[255, 255, 255]]])
	hsv1 = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
	print("BGR Value:", bgr)
	print("HSV Value:", hsv1)

	# compute the lower and upper limits
	lowerLimit = hsv1[0][0][0] - 10, 100, 100
	upperLimit = hsv1[0][0][0] + 10, 255, 255

	# display the lower and upper limits
	print("Lower Limit:", lowerLimit)
	print("Upper Limit", upperLimit)


def create_and_apply_mask():
	lowerLimit = np.array([-50, 0, 0])
	upperLimit = np.array([50, 255, 255])

	img = cv2.imread('first_frame.jpg')
	# Convert BGR to HSV
	# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# Create a mask. Threshold the HSV image to get only yellow colors
	# mask = cv2.inRange(hsv, lowerLimit, upperLimit)
	mask_new = cv2.imread('mask.jpg')
	masked_img = cv2.bitwise_and(img, mask_new)

	cv2.imshow('Mask image', masked_img)
	cv2.waitKey(0)
	cv2.imwrite('object.jpg', masked_img)

	opposite_mask = cv2.bitwise_not(mask_new)
	opposite_mask_img = cv2.bitwise_and(img, opposite_mask)
	cv2.imshow('Opposite Mask2', opposite_mask)
	cv2.waitKey(0)
	cv2.imshow('Opposite Mask on img', opposite_mask_img)
	cv2.waitKey(0)
	cv2.imwrite('background.jpg', opposite_mask_img)

	return masked_img, opposite_mask_img


if __name__ == '__main__':
	# new_frame()
	# find_colors()
	create_and_apply_mask()

# def make_mask():
#     # find hsv color boundaries
#     bgr = np.uint8([[[68, 163, 177]]])
#     hsv1 = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
#     print("BGR Value:", bgr)
#     print("HSV Value:", hsv1)
#
#     # compute the lower and upper limits
#     lowerLimit = hsv1[0][0][0] - 10, 100, 100
#     upperLimit = hsv1[0][0][0] + 10, 255, 255
#
#     # display the lower and upper limits
#     print("Lower Limit:", lowerLimit)
#     print("Upper Limit", upperLimit)
#
#     lowerLimit = np.array([16, 100, 100])
#     upperLimit = np.array([36, 255, 255])
#
#     img = cv2.imread('firstframe.png')
#     # Convert BGR to HSV
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     # Create a mask. Threshold the HSV image to get only yellow colors
#     mask = cv2.inRange(hsv, lowerLimit, upperLimit)
#     result = cv2.bitwise_and(img, img, mask=mask)
#
#     mask1 = np.zeros_like(img)
#     mask1 = np.zeros(img.shape[:2], np.uint8)
#     mask1[220:330, 150:250] = 255
#
#     mask2 = cv2.bitwise_and(mask1, mask, mask=mask1)
#     masked_img = cv2.bitwise_and(img, result, mask=mask1)
#
#     # cv2.imshow('Mask2', mask2)
#     # cv2.waitKey(0)
#     opposite_mask = cv2.bitwise_not(mask2)
#     opposite_mask_img = cv2.bitwise_and(img, img, mask=opposite_mask)
#     # cv2.imshow('Opposite Mask2', opposite_mask)
#     # cv2.waitKey(0)
#     # cv2.imshow('Opposite Mask on img',opposite_mask_img)
#     # cv2.waitKey(0)
#     cv2.imwrite('Opposite_Mask_on_img.jpg', opposite_mask_img)
#     # cv2.imshow('Mask1', mask1)
#     # cv2.waitKey(0)
#     # cv2.imshow('Mask', mask)
#     # cv2.waitKey(0)
#     # cv2.imshow('Masked Image', result)
#     # cv2.waitKey(0)
#     # cv2.imshow('Masked Image Final', masked_img)
#     # cv2.waitKey(0)
#
#     cv2.imwrite('Mask_on_balloon.jpg', masked_img)
#     return masked_img, opposite_mask_img
