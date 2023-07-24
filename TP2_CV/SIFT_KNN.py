import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img2 = cv.imread('im1.jpg',0)#test
img1 = cv.imread('rqt.png',0)#query


sift = cv.SIFT_create()

kp1 , des1 = sift.detectAndCompute(img1,None)
kp2 , des2 = sift.detectAndCompute(img2,None)

bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
print(matches)
print(matches[0][0].trainIdx)
print(matches[0][0].queryIdx)
print("pt")
print(kp2[matches[0][0].trainIdx].pt)
print(matches[0][0].distance)
print(matches[0][1].trainIdx)
print(matches[0][1].queryIdx)
print("pt")
print(kp2[matches[0][1].trainIdx].pt)
#print("dddd",10/3)
#print(matches[0][0].distance)
#print(matches[1][0].distance)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.2*n.distance:
        good.append([m])

good=np.array(good)
print("gooooof",good.shape)
print("ffffff",good[0][0].queryIdx)
print("fffffgggf",kp1[good[0][0].queryIdx].pt)
god = sorted(good[0],key=lambda x: x.distance)

# cv2.drawMatchesKnn expects list of lists as matches.
#print(good[1][0].distance)
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)
#plt.imshow(matches_knn)
plt.show()
