import cv2
import numpy as np
from julia.api import Julia

jl = Julia(compiled_modules=False)
from dpmmpythonStreaming.dpmmwrapper import DPMMPython


def new_frame():
	cap = cv2.VideoCapture('new_video.mp4')
	ret, frame = cap.read()
	cv2.imwrite('first_frame_new.jpg', frame)


def find_colors():
	# find hsv color boundaries
	bgr = np.uint8([[[241, 226, 230]]])
	hsv1 = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
	print("BGR Value:", bgr)
	print("HSV Value:", hsv1)

	# compute the lower and upper limits
	lowerLimit = hsv1[0][0][0] - 10, 100, 100
	upperLimit = hsv1[0][0][0] + 10, 255, 255

	# display the lower and upper limits
	print("Lower Limit:", lowerLimit)
	print("Upper Limit", upperLimit)


def create_mask():
	lowerLimit = np.array([-50, 0, 0])
	upperLimit = np.array([50, 255, 255])

	img = cv2.imread('first_frame_new.jpg')
	# Convert BGR to HSV
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# Create a mask. Threshold the HSV image to get only yellow colors
	mask = cv2.inRange(hsv, lowerLimit, upperLimit)
	mask_new = cv2.imread('mask_new.jpg')
	masked_img = cv2.bitwise_and(img, mask_new)

	# mask1 = np.zeros_like(img)
	# mask1 = np.zeros(img.shape[:2], np.uint8)
	# mask1[220:270, 150:200] = 255
	#
	# mask2 = cv2.bitwise_and(mask, mask, mask=mask1)
	# result = cv2.bitwise_and(img, img, mask=mask2)
	# cv2.imshow('Mask', mask)
	# cv2.waitKey(0)
	# cv2.imwrite('mask.jpg', mask)

	# cv2.imshow('Mask1', mask1)
	# cv2.waitKey(0)
	# cv2.imshow('Mask2', mask2)
	# cv2.waitKey(0)
	cv2.imshow('Mask image', masked_img)
	cv2.waitKey(0)
	cv2.imwrite('object.jpg', masked_img)

	# # cv2.imshow('Mask2', mask2)
	# # cv2.waitKey(0)
	opposite_mask = cv2.bitwise_not(mask_new)
	opposite_mask_img = cv2.bitwise_and(img, opposite_mask)
	cv2.imshow('Opposite Mask2', opposite_mask)
	cv2.waitKey(0)
	cv2.imshow('Opposite Mask on img', opposite_mask_img)
	cv2.waitKey(0)
	cv2.imwrite('background.jpg', opposite_mask_img)

	return masked_img, opposite_mask_img


def pre_process_image(img):
	dims = img.shape[0:2]
	histogram_arr = np.zeros((5, dims[0], dims[1]), dtype=np.float32)

	for x in range(dims[0]):
		for y in range(dims[1]):
			histogram_arr[0:3, x, y] = img[x][y]
			histogram_arr[3, x, y] = x / dims[0]
			histogram_arr[4, x, y] = y / dims[1]

	histogram_arr = np.reshape(histogram_arr, (5, dims[0] * dims[1]))

	return histogram_arr, dims


def color_image_by_labels(img, labels):
	color_map = {
		0: [255, 0, 0],  # Red
		1: [0, 255, 0],  # Green
		2: [0, 0, 255],  # Blue
		3: [255, 255, 0],  # Yellow
		4: [255, 0, 255],  # Magenta
		5: [0, 255, 255],  # Cyan
		6: [128, 0, 0],  # Maroon
		7: [0, 128, 0],  # Dark Green
		8: [0, 0, 128],  # Navy
		9: [128, 128, 128]  # Gray
	}

	rows, cols = np.shape(img)[0], np.shape(img)[1]
	new_img = np.zeros((rows, cols, 3))
	i = 0
	for row in range(rows):
		for col in range(cols):
			new_img[row][col] = color_map.get(labels[i])
			i += 1
	return new_img


def cluster_first_frame():
	background = cv2.imread('background.jpg')
	object = cv2.imread('object.jpg')
	object_data, _, = pre_process_image(object)
	background_data, _, = pre_process_image(background)

	prior = DPMMPython.create_prior(5, 0, 100, 100, 100)
	model_object = DPMMPython.fit_init(object_data, 10.0, prior=prior, iterations=60, init_clusters=1, verbose=True,
	                                   epsilon=0.001)
	model_background = DPMMPython.fit_init(background_data, 10.0, prior=prior, iterations=60, init_clusters=1,
	                                       verbose=True, epsilon=0.001)

	labels_background = DPMMPython.get_labels(model_background)
	labels_object = DPMMPython.get_labels(model_object)

	labeled_img_background = color_image_by_labels(background, labels_background)
	labeled_img_object = color_image_by_labels(object, labels_object)

	cv2.imwrite('background labeled.jpg', labeled_img_background)
	cv2.imshow("labeled img background", labeled_img_background)
	cv2.waitKey(0)
	cv2.imwrite('object labeled.jpg', labeled_img_object)
	cv2.imshow("labeled img object", labeled_img_object)
	cv2.waitKey(0)

	return model_object, model_background


if __name__ == '__main__':
	# cluster_first_frame()
	# new_frame()
	# find_colors()
	# create_mask()
	cluster_first_frame()


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
