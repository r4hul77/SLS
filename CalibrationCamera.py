#!/usr/bin/env python

import cv2
import numpy as np
import os
import json
import glob

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Defining the dimensions of checkerboard
CHECKERBOARD = (12, 17)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)*0.0254*0.25
print(objp)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob('/home/harsha/Desktop/SLS-CNH/Data/exp3/Calibration/V0/0/*.jpeg')
for i, fname in enumerate(images):
    if(i%20!=0):
        continue
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    cv2.imshow('img', img)
    cv2.waitKey(1)

cv2.destroyAllWindows()

h, w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
camera_props = {}
camera_props["ret"], camera_props["mtx"], camera_props["dist"], camera_props["rvecs"], camera_props["tvecs"] = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

with open("/home/harsha/Desktop/SLS-CNH/Data/exp3/Camera_20_0.0025_calib.json", "w") as f:
    json.dump(camera_props, f, cls=NumpyEncoder)

print("Camera matrix : \n")

