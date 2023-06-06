import cv2
import dpmmpythonStreaming.priors
from julia.api import Julia
from skimage.transform import resize

jl = Julia(compiled_modules=False)
from dpmmpythonStreaming.dpmmwrapper import DPMMPython
from dpmmpythonStreaming.priors import niw
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# code from git- example how to use DPMM
# data, gt = DPMMPython.generate_gaussian_data(10000, 2, 10, 100.0)

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

	# cv2.imshow('Mask2', mask2)
	# cv2.waitKey(0)
	opposite_mask = cv2.bitwise_not(mask2)
	opposite_mask_img = cv2.bitwise_and(img, img, mask=opposite_mask)
	# cv2.imshow('Opposite Mask2', opposite_mask)
	# cv2.waitKey(0)
	# cv2.imshow('Opposite Mask on img',opposite_mask_img)
	# cv2.waitKey(0)
	cv2.imwrite('Opposite_Mask_on_img.jpg', opposite_mask_img)
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
	# img = cv2.imread('firstframe.png')
	# object_masked_img, background_masked_img = make_mask()
	background_masked_img = cv2.imread('Opposite_Mask_on_img.jpg')
	object_masked_img = cv2.imread('Mask_on_balloon.jpg')
	object_data, _, = pre_process_image(object_masked_img)

	background_data, _, = pre_process_image(background_masked_img)

	# prior = DPMMPython.create_prior(5, mean_prior=object_data_mean, mean_str=1, cov_prior=object_data_cov, cov_str=1)
	# model = DPMMPython.fit_init(object_data, 1000.0, prior=prior, iterations=200, init_clusters=2,
	#                            verbose=True, burnout=5, gt=None, epsilon=0.0000001)
	prior = DPMMPython.create_prior(5, 0, 1, 1, 1)
	model_object = DPMMPython.fit_init(object_data, 1000.0, prior=prior, iterations=60, init_clusters=1, verbose=True,
	                                   epsilon=0.000001)
	model_background = DPMMPython.fit_init(background_data, 1000.0, prior=prior, iterations=60, init_clusters=1,
	                                       verbose=True, epsilon=0.000001)
	labels_background = DPMMPython.get_labels(model_background)
	labels_object = DPMMPython.get_labels(model_object)
	labeled_img_background = color_image_by_labels(background_masked_img, labels_background)
	labeled_img_object = color_image_by_labels(object_masked_img, labels_object)
	cv2.imshow("labeled img background", labeled_img_background)
	cv2.waitKey(0)
	cv2.imshow("labeled img object", labeled_img_object)
	cv2.waitKey(0)

	return model_object, model_background


# Function to turn a image into a 5D array of RGBXY
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


def jjjj():
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture('video.mp4')
    model_object, model_background = cluster_first_frame()

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            # cv2.imshow('Frame', frame)
            frame_data = pre_process_image(frame)
            model_frame = DPMMPython.fit_partial(model_object, 1, 1, frame_data)
            frame_labels = DPMMPython.get_labels(model_frame)
            colored_frame = color_image_by_labels(frame, frame)
            cv2.imshow('Labeled Frame', colored_frame)
            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break

    # When everything is done, release the video capture object
    cap.release()






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
	#cluster_first_frame()
	jjjj()


