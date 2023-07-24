import cv2
import numpy as np




def mouseHandler(event, x, y, flags, data):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(data['im'], (x, y), 3, (0, 255, 255), 5,cv2.LINE_AA)
            cv2.imshow("Image", data['im'])
            if len(data['points']) < 4:
                data['points'].append([x, y])


#Une fonction qui retourne la liste des point selectionnées
def get_four_points(im):
        # Initialiser les données a envoyer pour mousehandler
        data = {}
        data['im'] = im.copy()
        data['points'] = []
        cv2.imshow("Image", im)
        cv2.setMouseCallback("Image", mouseHandler, data)
        cv2.waitKey(0)
        points = np.vstack(data['points']).astype(float)
        return points

# Lire l'image qu'on veut ajouter
im_src = cv2.imread('IMG.jpg')
size = im_src.shape

# Creation d'in vecteur des points
pts_src = np.array(
        [
            [0, 0],
            [size[1] - 1, 0],
            [size[1] - 1, size[0] - 1],
            [0, size[0] - 1]
        ], dtype=float
    )

# Lecture de l'image originale
im_dst = cv2.imread('book1.jpg')

# Recuperer les 4 point de cette image
pts_dst = get_four_points(im_dst)

# Le calcule del'homography entre ses points
h, status = cv2.findHomography(pts_src, pts_dst)

# Warp source image
im_temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
cv2.imshow("Image WARPED", im_temp)
cv2.waitKey(0)
# colorier la zone selectionné (par les 4point ) en noir
cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, cv2.LINE_AA)
# additionner lesimages
im_dst = im_dst + im_temp

#Afficher la nouvelle image
cv2.imshow("Image", im_dst)

cv2.waitKey(0)