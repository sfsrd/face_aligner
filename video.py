from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import numpy as np

def bb_to_rect(bb):
    top=bb[1]
    left=bb[0]
    right=bb[0]+bb[2]
    bottom=bb[1]+bb[3]
    return np.array([top, right, bottom, left]) 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)


cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 2)
    for rect in rects:
    	(x, y, w, h) = rect_to_bb(rect)
    	faceOrig = imutils.resize(frame[y:y + h, x:x + w], width=256)
    	faceAligned = fa.align(frame, gray, rect)
    	faceAligned = fa.align(frame, gray, rect)
    	cv2.imshow("Original", faceOrig)
    	cv2.imshow("Aligned", faceAligned)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


