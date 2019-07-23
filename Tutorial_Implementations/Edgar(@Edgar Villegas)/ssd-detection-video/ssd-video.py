# Code based on tutorial from pyimagesearch.net (adapted by Edgar - edgarinvillegas@hotmail.com). v1.0
# USAGE
# python ssd-video.py --input videos/dog-kid.mp4

# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
	"diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("model/MobileNetSSD_deploy.prototxt.txt", "model/MobileNetSSD_deploy.caffemodel")

# initialize the video stream, pointer to output video file, and frame dimensions
vs = cv2.VideoCapture(args["input"])

def processFrame(image):
	image = imutils.resize(image, width=400)

	# grab the frame dimensions and convert it to a blob
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
	##
	# pass the blob through the network and obtain the detections and predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	# pass the blob through the network and obtain the detections and predictions
	# print("[INFO] computing object detections...")
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the `detections`,
			# then compute the (x, y)-coordinates of the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# display the prediction
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			print("[INFO] {}".format(label))
			cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
	return image

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end of the stream
	if not grabbed: break

	frame = processFrame(frame)

	cv2.imshow('win', frame)
	key = cv2.waitKey(1) & 0xFF

# release resources
print("[INFO] cleaning up...")
vs.release()