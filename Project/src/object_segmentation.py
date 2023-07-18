import cv2
from julia.api import Julia
from scipy.stats import multivariate_normal
from dpmmpythonStreaming.dpmmwrapper import DPMMPython
import numpy as np
jl = Julia(compiled_modules=False)


def color_image_by_labels(img, labels):
    color_map = {
        0: [0, 0, 128],  # Navy
        1: [255, 0, 0],  # Red
        2: [0, 255, 0],  # Green
        3: [0, 0, 255],  # Blue
        4: [255, 255, 0],  # Yellow
        5: [255, 0, 255],  # Magenta
        6: [0, 255, 255],  # Cyan
        7: [128, 0, 0],  # Maroon
        8: [0, 128, 0],  # Dark Green

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
    model_object = DPMMPython.fit_init(object_data, 100.0, prior=prior, iterations=50, init_clusters=1, verbose=True,
                                       epsilon=0.001)
    model_background = DPMMPython.fit_init(background_data, 100.0, prior=prior, iterations=50, init_clusters=1,
                                           verbose=True, epsilon=0.001)

    # labels_background = DPMMPython.get_labels(model_background)
    # labels_object = DPMMPython.get_labels(model_object)
    #
    # labeled_img_background = color_image_by_labels(background, labels_background)
    # labeled_img_object = color_image_by_labels(object, labels_object)
    #
    # cv2.imshow("labeled img background", labeled_img_background)
    # cv2.waitKey(0)
    # cv2.imwrite('background labeled.jpg', labeled_img_background)
    # cv2.imshow("labeled img object", labeled_img_object)
    # cv2.waitKey(0)
    # cv2.imwrite('object labeled.jpg', labeled_img_object)

    return model_object, model_background


# Function to turn an image into a 5D array of RGBXY
def pre_process_image(img):
    dims = img.shape[0:2]
    histogram_arr = np.zeros((5, dims[0], dims[1]), dtype=np.float32)

    for x in range(dims[0]):
        for y in range(dims[1]):
            histogram_arr[0:3, x, y] = img[x][y]
            histogram_arr[3, x, y] = x / 100
            histogram_arr[4, x, y] = y / 100

    histogram_arr = np.reshape(histogram_arr, (5, dims[0] * dims[1]))

    return histogram_arr, dims


def extract_model_means_vars(model):
    model_means = []
    model_vars = []
    for i in range(len(model.group.local_clusters)):
        model_means.append(model.group.local_clusters[i].cluster_params.cluster_params.distribution.μ)
        model_vars.append(model.group.local_clusters[i].cluster_params.cluster_params.distribution.Σ)
    return model_means, model_vars


def is_object_pixel(pixel_data, object_means, object_vars, background_means, background_vars):
    object_pixel = False
    max_prob = 0
    for i in range(len(object_means)):
        mean = object_means[i]
        var = object_vars[i]
        prob = multivariate_normal.pdf(x=pixel_data, mean=mean, cov=var)
        if prob > max_prob and np.abs(pixel_data[3] - mean[3]) < 1.2 and np.abs(pixel_data[4] - mean[4]) < 2:
            max_prob = prob
            object_pixel = True
    if not object_pixel:
        return object_pixel
    for j in range(len(background_means)):
        mean = background_means[j]
        var = background_vars[j]
        prob = multivariate_normal.pdf(x=pixel_data, mean=mean, cov=var)
        if prob > max_prob:
            return False

    return object_pixel


def video_segmentation():
    cap = cv2.VideoCapture('input_video.mp4')
    object_model, background_model = cluster_first_frame()
    object_means, object_vars = extract_model_means_vars(object_model)
    background_means, background_vars = extract_model_means_vars(background_model)
    object_color = [255, 0, 255]  # Magenta

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # x
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # y
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc=fourcc, fps=fps, frameSize=(frame_width, frame_height))

    # Read until video is completed
    t = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        colored_object_image = np.zeros_like(frame)
        new_object_data = []
        new_background_data = []
        t += 1
        print(t)

        if ret == True:
            for x in range(frame.shape[0]):
                for y in range(frame.shape[1]):
                    pixel_data = np.zeros(5, dtype=np.float32)
                    pixel_data[0:3] = frame[x][y]
                    pixel_data[3] = x / 100
                    pixel_data[4] = y / 100

                    if is_object_pixel(pixel_data, object_means, object_vars, background_means, background_vars):
                        new_object_data.append(pixel_data)
                        colored_object_image[x][y] = object_color
                    else:
                        new_background_data.append(pixel_data)
                        colored_object_image[x][y] = frame[x][y]  # original color
            alpha = 0.12  # Adjust the transparency (0.0 - fully transparent, 1.0 - fully opaque)
            output_frame = cv2.addWeighted(colored_object_image, alpha, frame, 1 - alpha, 0)

            # Display the resulting frame
            cv2.imshow('Segmented Video', output_frame)
            out.write(output_frame)

            new_object_data = np.reshape(new_object_data, (5, len(new_object_data)))
            new_background_data = np.reshape(new_background_data, (5, len(new_background_data)))
            object_model = DPMMPython.fit_partial(object_model, iterations=3, t=t, data=new_object_data)
            background_model = DPMMPython.fit_partial(background_model, iterations=3, t=t, data=new_background_data)

        else:
            break
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # When everything is done, release the video capture object
    cap.release()
    out.release()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    video_segmentation()
