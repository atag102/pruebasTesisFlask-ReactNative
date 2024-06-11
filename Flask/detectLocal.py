import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import os

cat_face_cascade = cv2.CascadeClassifier('./haarcascade_frontalcatface_extended.xml')
carpeta_imagenes = './AllFaceDataset/0030'
imagenes = os.listdir(carpeta_imagenes)

def cara__recortada(img):
    cats = cat_face_cascade.detectMultiScale(img, 1.13, 1)
    print(cats)
    for (x, y, w, h) in cats:
        img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img

for imagen in imagenes:
    image_path = os.path.join(carpeta_imagenes, imagen)
    img = mpimg.imread(image_path)
    """ print("MOSTRANDO IMAGEN ORIGINAL")
    plt.imshow(img)
    plt.show() """

    print(f"Mostrando resultado para la imagen {imagen}")
    img_recortada=cara__recortada(img)
    plt.imshow(img_recortada)
    plt.show()














img = mpimg.imread('../Flask/AllFaceDataset/0029/0029_001'+'.JPG',3)

