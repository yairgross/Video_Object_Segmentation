import cv2
from julia.api import Julia

jl = Julia(compiled_modules=False)
from dpmmpythonStreaming.dpmmwrapper import DPMMPython
from dpmmpythonStreaming.priors import niw
import numpy as np
import cv2 as cv


# code from git- example how to use DPMM
#data, gt = DPMMPython.generate_gaussian_data(10000, 2, 10, 100.0)

# batch1 = data[:, 0:5000]
# batch2 = data[:, 5000:]
# prior = DPMMPython.create_prior(2, 0, 1, 1, 1)
# model = DPMMPython.fit_init(batch1, 100.0, prior=prior, verbose=True, burnout=5, gt=None, epsilon=0.0000001)
# labels = DPMMPython.get_labels(model)
# model = DPMMPython.fit_partial(model, 1, 2, batch2)
# labels = DPMMPython.get_labels(model)
# print(labels)

def make_mask():
	# find hsv color boundaries
	bgr = np.uint8([[[68, 163, 177]]])
	hsv1 = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
	print("BGR Value:", bgr)
	print("HSV Value:", hsv1)

	# compute the lower and upper limits
	lowerLimit = hsv1[0][0][0] - 10, 100, 100
	upperLimit = hsv1[0][0][0] + 10, 255, 255

	# display the lower and upper limits
	print("Lower Limit:", lowerLimit)
	print("Upper Limit", upperLimit)

	lowerLimit = np.array([16, 100, 100])
	upperLimit = np.array([36, 255, 255])

	img = cv2.imread('firstframe.png')
	# Convert BGR to HSV
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# Create a mask. Threshold the HSV image to get only yellow colors
	mask = cv2.inRange(hsv, lowerLimit, upperLimit)
	result = cv2.bitwise_and(img, img, mask=mask)

	mask1 = np.zeros_like(img)
	mask1 = np.zeros(img.shape[:2], np.uint8)
	mask1[220:330, 150:250] = 255

	mask2 = cv2.bitwise_and(mask1, mask, mask=mask1)
	masked_img = cv2.bitwise_and(img, result, mask=mask1)

	#cv2.imshow('Mask2', mask2)
	#cv2.waitKey(0)
	opposite_mask = cv2.bitwise_not(mask2)
	opposite_mask_img = cv2.bitwise_and(img, img, mask=opposite_mask)
	# cv2.imshow('Opposite Mask2', opposite_mask)
	# cv2.waitKey(0)
	# cv2.imshow('Opposite Mask on img',opposite_mask_img)
	# cv2.waitKey(0)
	cv2.imwrite('Opposite_Mask_on_img.jpg',opposite_mask_img)
	# cv2.imshow('Mask1', mask1)
	# cv2.waitKey(0)
	# cv2.imshow('Mask', mask)
	# cv2.waitKey(0)
	# cv2.imshow('Masked Image', result)
	# cv2.waitKey(0)
	# cv2.imshow('Masked Image Final', masked_img)
	# cv2.waitKey(0)

	cv2.imwrite('Mask_on_balloon.jpg', masked_img)
	return masked_img, opposite_mask_img


def cluster_first_frame():
	img = cv2.imread('firstframe.png')
	object_masked_img, background_masked_img = make_mask()
	masked_img_data = np.array(object_masked_img)
	rows, cols, rgb = masked_img_data.shape
	x = masked_img_data.reshape(rows * cols, rgb)
	prior = DPMMPython.create_prior(2, 0, 1, 1, 1)
	batch1 = x[:, 0:5000]
	model = DPMMPython.fit_init(batch1, 10.0, prior=prior, iterations= 10, burnout=5)
	labels = DPMMPython.get_labels(model)
	print(labels)






	# def jjjj():
	# Create a VideoCapture object and read from input file
	# If the input is the camera, pass 0 instead of the video file name
	#cap = cv2.VideoCapture('video.mp4')


# Check if camera opened successfully
# if (cap.isOpened() == False):
#     print("Error opening video stream or file")

# # Read until video is completed
# while (cap.isOpened()):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if ret == True:
#
#         # Display the resulting frame
#
#         cv2.imshow('Frame', frame)
#
#         # Press Q on keyboard to exit
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#
#     # Break the loop
#     else:
#         break
#
# # When everything done, release the video capture object
# cap.release()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
	cluster_first_frame()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
