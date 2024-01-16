from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
#import RPi.GPIO as GPIO
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

#GPIO.setmode(GPIO.BOARD)
#GPIO.setup(5, GPIO.OUT)
#GPIO.setup(3, GPIO.OUT)
#GPIO.setup(7, GPIO.OUT)
#GPIO.setup(8, GPIO.OUT)


def mouth_aspect_ratio(mouth):
	
	A = dist.euclidean(mouth[2], mouth[10]) 
	B = dist.euclidean(mouth[3], mouth[8]) 

	
	C = dist.euclidean(mouth[0], mouth[6]) 

	
	mar = (A + B) / (2.0 * C)

	
	return mar


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False, default='shape_predictor_68_face_landmarks.dat',
	help="path to facial landmark predictor")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())


MOUTH_AR_THRESH = 0.7

print("[INFO] Loading Facial Landmark Predictor....")
print("Fetching frames from RaspBerry Pi Cam V2....")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(mStart, mEnd) = (49, 68)

print("[INFO] Starting Video Stream Thread...")
print("Capturing Frames: ")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

frame_width = 600
frame_height = 300

out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
time.sleep(1.0)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 0)

	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		mouth = shape[mStart:mEnd]

		mouthMAR = mouth_aspect_ratio(mouth)
		mar = mouthMAR
		mouthHull = cv2.convexHull(mouth)
		
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		cv2.putText(frame, "MAR: {:.2f}".format(mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		if mar > MOUTH_AR_THRESH:
			cv2.putText(frame, "Mouth is Open!", (30,60),
	       	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
			cv2.putText(frame, "SYSTEM IS ON...", (30,85),
	       	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
#			GPIO.output(3, True)
			#GPIO.output(7, True)
			print("System On")
			
		else: #mar < MOUTH_AR_THRESH:
			print("Stop")			
# 			 GPIO.output(3, False)
			#GPIO.output(7, False)

	out.write(frame)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()


#			out.write(frame)
#	cv2.imshow("Frame", frame)
#	key = cv2.waitKey(1) & 0xFF
#
#	if key == ord("q"):
#		break
#
#cv2.destroyAllWindows()
#vs.stop()