# import computer vision library(cv2) in this code
import cv2

img_path = "im1.jpg"
image_c = cv2.imread(img_path)
image = cv2.cvtColor(image_c,cv2.COLOR_BGR2GRAY)
#image=image_c

# mentioning absolute path of the image
#cv2.GaussianBlur(src, ksize, sigmaX, sigmaY, borderType)
#borderType: This specify boundaries of an image while kernel is applied on borders of an image.
# cv2.BORDER_DEFAULT: gfedcb|abcdefgh|gfedcba

#blur_img = cv2.GaussianBlur(image,(3,3),6.0,0, cv2.BORDER_DEFAULT)
blur_img=image
n=20

for i in range(n):
    blur_img = cv2.GaussianBlur(blur_img, (3, 3), 6.0, 0, cv2.BORDER_DEFAULT)


    # show the image on the newly created image window
	#cv2.imshow('Blur image',blur_img)
cv2.imwrite("blur_imag1.png", blur_img)


#Question 2
sig1=2.0
sig2=3.0

blur1 = cv2.GaussianBlur(image, (3, 3), sig1, 0, cv2.BORDER_DEFAULT)
blur1 = cv2.GaussianBlur(blur1, (3, 3), sig2, 0, cv2.BORDER_DEFAULT)

sig=20.0
blur2 = cv2.GaussianBlur(image, (3, 3), sig, 0, cv2.BORDER_DEFAULT)




while(True):
    #Affichage question1
    #cv2.imshow("im1",image_c)
    cv2.imshow("im1 gray", image)
    #cv2.imshow("gauss", blur_img)


    #Affichage question2
    cv2.imshow("gauss1", blur1)
    cv2.imshow("gauss2", blur2)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cv2.waitKey(1)
cv2.destroyAllWindows()
#cv2.waitKey(0)
