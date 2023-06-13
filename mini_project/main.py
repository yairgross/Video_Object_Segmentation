import pandas as pd
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

def cluster_object_in_frame(frame, object_model, background_model):
    #TODO: implement to return: list of pixels in object, new_object_model, new_background_model
    object_mask = cv2.imread('Mask_on_balloon.jpg')

    # Define the range of x and y values
    x_values = np.arange(220, 330)  # Example: from 1 to 4
    y_values = np.arange(150, 250)  # Example: from 6 to 9

    # Create a meshgrid of x and y values
    xx, yy = np.meshgrid(x_values, y_values)

    # Convert the meshgrid to a list of (x, y) pairs
    xy_pairs = list(zip(xx.ravel(), yy.ravel()))

    return xy_pairs, object_model, background_model


def main(video_name):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(video_name)
    #object_model, background_model = cluster_first_frame()
    object_color = [0, 255, 0]  # Green color

    iteration = 0
# Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc=fourcc, fps=20.0, frameSize=(frame_width, frame_height))

    # Read until video is completed
    while cap.isOpened():
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            # cv2.imshow('Frame', frame)
            frame_data = pre_process_image(frame)
            # # model_frame = DPMMPython.fit_partial(object_model, iterations=1, t=iteration, data=frame_data)
            # # frame_labels = DPMMPython.get_labels(model_frame)
            # # colored_frame = color_image_by_labels(frame, frame)
            # cv2.imshow('Labeled Frame', colored_frame)
            iteration += 1
            print(iteration)
            mask1 = np.zeros_like(frame)
            mask1 = np.zeros(frame.shape[:2], np.uint8)
            mask1[220:330, 150:250] = 255

            colored_image = np.zeros_like(frame)
            for x in range(mask1.shape[0]):
                for y in range(mask1.shape[1]):
                    if mask1[x][y] != 0:  # Object pixel
                        colored_image[x][y] = object_color
            alpha = 0.15  # Adjust the transparency (0.0 - fully transparent, 1.0 - fully opaque)
            output_frame = cv2.addWeighted(colored_image, alpha, frame, 1 - alpha, 0)

            # Display the resulting frame
            cv2.imshow('Segmented Video', output_frame)
            out.write(output_frame)
            #cv2.waitKey(0)

        # # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        # # Break the loop
        # else:
        #     break

    # When everything is done, release the video capture object
    cap.release()
    out.release()

def first_frame(name):
    cap = cv2.VideoCapture(name)
    ret, frame = cap.read()
    cv2.imwrite('first_frame.jpg', frame)
    return frame.shape

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # cluster_first_frame()
    main('new_video.mp4')
