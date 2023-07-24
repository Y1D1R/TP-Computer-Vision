import cv2

#creer une liste des images a donner pour la fonction stitch
images = []
size = 400

#Lire les images et les stockées dans la liste
img1 = cv2.imread('photo2.jpg')
cv2.imshow("image droite",img1)
cv2.waitKey(0)
images.append(img1)

img2 = cv2.imread('photo1.jpg')
cv2.imshow("image gauche",img2)
cv2.waitKey(0)
images.append(img2)


#creer un Objet Stitcher
imageStitcher = cv2.Stitcher_create()

erreur , image_finale = imageStitcher.stitch(images)

if not erreur:

    cv2.imshow("image finale",image_finale)
    cv2.waitKey(0)

