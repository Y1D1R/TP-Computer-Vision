import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('im1.jpg',0)
img2 = cv.imread('rqt.png',0)

sift = cv.SIFT_create()

kp1 , des1 = sift.detectAndCompute(img1,None)
kp2 , des2 = sift.detectAndCompute(img2,None)

BFMatcher = cv.BFMatcher(cv.NORM_L1,crossCheck=True)

matches = BFMatcher.match(des1,des2)
#print("nb = ",len(matches))


#Trier dans l'ordre de leur distances
matches = sorted(matches, key = lambda x:x.distance)

print(matches[0].distance)

print(matches[1].distance)
print(matches[2].distance)
#dessiner les 15 grandes distances
img_matched = cv.drawMatches(img1,kp1,img2,kp2,matches[:15],img2,flags=4)

plt.imshow(img_matched)
plt.show()
