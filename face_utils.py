import math
from imutils import face_utils
from imutils.face_utils.helpers import FACIAL_LANDMARKS_IDXS
import numpy as np
import imutils
import dlib
import cv2
from scipy.spatial import distance as dist

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

MOUTH_AR_THRESH = 0.7

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def get_rotated_mouth_loc_with_height(image):
	x1,y1,w1,h1,h_in,y = 1,1,1,1,1,1
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)
	if(len(rects) > 0):
		for (i, rect) in enumerate(rects):
			shape = predictor(gray, rect)
			shape = shape_to_np(shape)
			x_lowest_in_face, y_lowest_in_face = shape[9]
			for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
				if(name == "mouth"):
					mouth_rect = cv2.minAreaRect(np.array([shape[i:j]]))
				if(name == "inner_mouth"):
					inner_mouth_rect = cv2.minAreaRect(np.array([shape[i:j]]))
		return {"mouth_rect":mouth_rect,
		"inner_mouth_rect":inner_mouth_rect, "image_ret":image, "y_lowest_in_face":y_lowest_in_face, "shape": shape}
	else:
		return {"error":"true", "message":"No Face Found!"}


def get_mouth_loc_with_height(image):
	x1,y1,w1,h1,h_in,y = 1,1,1,1,1,1
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)
	if(len(rects) > 0):
		for (i, rect) in enumerate(rects):
			shape = predictor(gray, rect)
			shape = shape_to_np(shape)
			x_lowest_in_face, y_lowest_in_face = shape[9]
			for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
				if(name == "mouth"):
					(x1, y1, w1, h1) = cv2.boundingRect(np.array([shape[i:j]]))
				if(name == "inner_mouth"):
					(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
					h_in = h
		return {"mouth_x":x1,
		"mouth_y":y1,"mouth_w":w1,"mouth_h":h1, "image_ret":image, "height_of_inner_mouth":h_in, 
		"inner_mouth_y":y, "y_lowest_in_face":y_lowest_in_face, "shape": shape}
	else:
		return {"error":"true", "message":"No Face Found!"}
							

def get_mouth_loc(image):
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)

	for (i, rect) in enumerate(rects):
		shape = predictor(gray, rect)
		shape = shape_to_np(shape)
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			if(name == "mouth"):
				clone = image.copy()
				cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
				for (x, y) in shape[i:j]:
					cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
				(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
				return x,y,w,h, image
			
def draw_mouth(image, shape):
	(j, k) = FACIAL_LANDMARKS_IDXS["mouth"]
	pts_mouth = shape[j:k]
	for (x, y) in pts_mouth:
		cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
	
	return image

def point_on_lower_lip(x, y, shape, x1, x2, y1, y2):
	x = math.floor((x*x1)/x2)
	y = math.floor((y*y1)/y2)
	lower_lip_points = [48, 50, 59, 60, 58, 67, 57, 66, 56, 65, 55, 64, 54]
	pts = np.array([shape[i] for i in lower_lip_points])
	result = cv2.pointPolygonTest(pts, (x,y), measureDist=False)
	print(result)
	if(result>=0) :
		return True
	else:
		return False


def mouth_aspect_ratio(shape):
	(mStart, mEnd) = (49, 68)
	mouth = shape[mStart:mEnd] 
	A = dist.euclidean(mouth[2], mouth[10])
	B = dist.euclidean(mouth[4], mouth[8]) 

	C = dist.euclidean(mouth[0], mouth[6]) 

	mar = (A + B) / (2.0 * C)

	return mar
	
