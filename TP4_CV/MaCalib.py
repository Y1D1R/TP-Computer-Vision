#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob


# Defining the dimensions of checkerboard
CHECKERBOARD = (7, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.001)
# max number of iterations=30

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []
error = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

nx = 9
# Enter the number of inside corners in y
ny = 7
# Extracting path of individual image stored in a given directory
print("edd")
images = glob.glob('./MES IMAGES/*.jpg')
print("zzzzzzzz")
for fname in images:
    img = cv2.imread(fname)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

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
    cv2.waitKey(0)

cv2.destroyAllWindows()

h, w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("ret: \n")
print(ret)
print("Camera matrix (matrice intrinseques) : \n")
print(mtx)
print("la distance focale(nb pix/mm) Fx = ",mtx[0][0],"  Fy = ",mtx[1][1])
print("les coordonnees du point d'impact sont Ox= ",mtx[0][2],"  Oy = ",mtx[1][2])
print("\n\n")
print("dist (les coefficients de distortion) : \n")
print(dist)
print("rvecs (le vecteur de rotation) : \n")
print(rvecs)
print("tvecs (le vecteur de translation) : \n")
print(tvecs)

# Save parameters into numpy file
np.save("./camera_params/ret", ret)
np.save("./camera_params/mtx", mtx)
np.save("./camera_params/dist", dist)
np.save("./camera_params/rvecs", rvecs)
np.save("./camera_params/tvecs", tvecs)

