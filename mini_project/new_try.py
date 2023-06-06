import cv2


def new_frame():
	cap = cv2.VideoCapture('bear video.mp4')
	ret, frame = cap.read()
	cv2.imwrite('first_frame_bear.jpg', frame)


if __name__ == '__main__':
	# cluster_first_frame()
	new_frame()
