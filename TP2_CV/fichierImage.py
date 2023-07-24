import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

queries = [cv.imread(file) for file in glob.glob("C:\Espace_Python\humanBodyFallDetection-master\TAI\TP2_CV\queries\*.png")]
licence = [cv.imread(file) for file in glob.glob("C:\Espace_Python\humanBodyFallDetection-master\TAI\TP2_CV\License\*.jpg")]

#img2=queries[3]
queries = np.array(queries)

def Sift_detector(query,licence):
  #Lecture de l'image qui contient la plaque d'immatriculation
  img2=query

  #Creation de l'objet SIFT
  sift = cv.SIFT_create()

  #RGB To Gray scale
  gray_img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

  #La detection des point d'interets et les descripteurs
  kp2, des2 = sift.detectAndCompute(gray_img2, None)

  #nbm est une variable qui va contenir le nombre des points qui ont une meilleure correspondance
  #nbm = 0

  #img3 va contenir l'image finale
  img3 = img2


  d = False

  licence = np.array(licence)
  #parcourir tout l'ensemble des images des voitures et faire une comparaison a chaque fois
  for i in range(licence.shape[0]):
      img1 = licence[i]
      gray_img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

      kp1, des1 = sift.detectAndCompute(gray_img1, None)

      BFMatcher = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

      matches = BFMatcher.match(des1, des2)
      # print("nb = ",len(matches))

    # Trier dans l'ordre de leur distances
      matches = sorted(matches, key=lambda x: x.distance)
      #print(matches[0].distance)

      dis_calcule=min(matches,key=lambda x: x.distance).distance
      img_matched = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], img2, flags=4)
      if(d == False):
          d_min = dis_calcule
          d=True
      else:
          if (dis_calcule <= d_min):
              #nbm = len(matches)
              d_min=dis_calcule
              img3 = img_matched

  plt.imshow(img3)
  plt.show()


for i in range(queries.shape[0]):
    Sift_detector(queries[i],licence)