import cv2 as cv


#Lire l'image
img1 = cv.imread('im1.jpg')
img2 = cv.imread('rqt.png')

gray_img1 = cv.cvtColor(img1,cv.COLOR_RGB2GRAY)
gray_img2 = cv.cvtColor(img2,cv.COLOR_RGB2GRAY)

#Point d'interet (KeyPoints)

sift = cv.SIFT_create()
kp1,des1 = sift.detectAndCompute(gray_img1,None)
kp2,des2 = sift.detectAndCompute(gray_img2,None)

print(kp1[0].pt)
print(kp1[0].size)
print(kp1[0].angle)
print(kp1[0].response)
print(des1[0])
print(len(des1[0]))


img1_sift = cv.drawKeypoints(gray_img1,kp1,img1,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_sift = cv.drawKeypoints(gray_img2,kp2,img2,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


cv.imshow("SIFT IMAGE1", img1_sift)
cv.imshow("IMAGE2", img2_sift)
cv.waitKey(0)
cv.destroyAllWindows()