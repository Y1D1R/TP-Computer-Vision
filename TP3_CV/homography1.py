import cv2
import numpy as np

def mouseHandler(event,x,y,flags,param):
    global im_temp, pts_src

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(im_temp,(x,y),3,(0,255,255),5,cv2.LINE_AA)
        cv2.imshow("Image", im_temp)
        if len(pts_src) < 4:
        	pts_src = np.append(pts_src,[(x,y)],axis=0)


# Read in the image.
im_src = cv2.imread("book1.jpg")

# Destination image
height, width = 400, 600
im_dst = np.zeros((height,width,3),dtype=np.uint8)


# Create a list of points.
pts_dst = np.empty((0,2))
pts_dst = np.append(pts_dst, [(0,0)], axis=0)
pts_dst = np.append(pts_dst, [(width-1,0)], axis=0)
pts_dst = np.append(pts_dst, [(width-1,height-1)], axis=0)
pts_dst = np.append(pts_dst, [(0,height-1)], axis=0)

# Create a window
cv2.namedWindow("Image", 1)

im_temp = im_src
pts_src = np.empty((0,2))
# Appeler evenement de la souris
cv2.setMouseCallback("Image",mouseHandler)


cv2.imshow("Image", im_temp)
cv2.waitKey(0)

#Calculer homographhy selon les points pour trouver la transformation de perspective
#pts_src and pts_dst sont des vecteur numpy des points dans source et dest image, on a besoin d'au moin 4 ptn

#On utilise RANSAC pour eliminer au mieux les faux appariements(OUTLIERS)
tform, status = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC,maxIters=1000)


# Appliquer une deformation
im_dst = cv2.warpPerspective(im_src, tform,(width,height),borderMode=cv2.BORDER_REPLICATE)

print(pts_dst)
#Afin de mettre la correspondance
print("La matrice d'homographie = \n", tform)
print("\nstatus = \n", status)

cv2.imshow("Image", im_dst)
cv2.imwrite("out.png", im_dst)
cv2.waitKey(0)
