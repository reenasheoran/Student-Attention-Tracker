# Importing OpenCV Library for basic image processing functions
import cv2
# Numpy for array related functions
import numpy as np
import face_recognition
# Dlib for deep learning based Modules and face landmark detection
import dlib
# face_utils for basic operations of conversion
from imutils import face_utils

import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0: 7, 0: 6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('data/images/*.jpg')
#print(images)

for image in images:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
        #cv2.imshow('img', img)
        cv2.waitKey(500)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print(mtx)
        print(dist)

cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)

# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def ref3DModel():
    modelPoints = [[0.0,0.0,0.0],
                   [0.0,-330.0,-65.0],
                   [-255.0,170.0,-135.0],
                   [225.0,170.0,-135.0],
                   [-150.0,-150.0,-125.0],
                   [150.0,-150.0,-125.0]]
    return np.array(modelPoints,dtype=np.float64)

def ref2DImagePoints(shape):
    imagePoints = [[shape.part(30).x,shape.part(30).y],
                   [shape.part(8).x,shape.part(8).y],
                   [shape.part(36).x,shape.part(36).y],
                   [shape.part(45).x,shape.part(45).y],
                   [shape.part(48).x,shape.part(48).y],
                   [shape.part(54).x,shape.part(54).y]]
    return np.array(imagePoints,dtype=np.float64)

def drawPolyline(img,shapes,start,end,isClosed=False):
    points = []
    for i in range(start,end+1):
        point = [shapes.part(i).x,shapes.part(i).y]
        points.append(point)
    points = np.array(points,dtype=np.float32)
    cv2.polylines(img,np.int32([points]),isClosed,(255,80,0),thickness=1,lineType=cv2.LINE_8)

def draw(img,shapes):
    drawPolyline(img, shapes, 0, 16)
    drawPolyline(img, shapes, 17, 21)
    drawPolyline(img, shapes, 22, 26)
    drawPolyline(img, shapes, 27, 30)
    drawPolyline(img, shapes, 30, 35, True)
    drawPolyline(img, shapes, 36, 41, True)
    drawPolyline(img, shapes, 42, 47, True)
    drawPolyline(img, shapes, 48, 59, True)
    drawPolyline(img, shapes, 60, 67, True)

while True:
    GAZE = "Face Not Found"
    ret,img = cap.read()
    
    faces = detector(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),0)
    face3Dmodel = ref3DModel()
    
    for face in faces:
        shape = predictor(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),face)
        
        draw(img,shape)
        
        refImgPts = ref2DImagePoints(shape)
        
        height,width,channels = img.shape

        mdists = np.zeros((4,1),dtype=np.float64)
        
        success,rotationVector,translationVector = cv2.solvePnP(face3Dmodel,refImgPts,mtx,mdists)
        
        noseEndPoints3D = np.array([[0,0,1000.0]],dtype=np.float64)
        noseEndPoint2D,jacobian = cv2.projectPoints(noseEndPoints3D, rotationVector, translationVector, mtx, mdists)
        
        rmat, jac = cv2.Rodrigues(rotationVector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        
        if angles[1] < -15:
            GAZE = "Looking: Left"
        elif angles[1] > 15:
            GAZE = "Looking: Right"
        else:
            GAZE = "Forward"
            
    cv2.putText(img,GAZE,(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,80),2)
    cv2.imshow("Head Pose",img)
    
    key = cv2.waitKey(10) & 0xFF
    if key == 27:
        break
        
cap.release()
cv2.destroyAllWindows()