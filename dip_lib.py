import cv2
import numpy as np
from matplotlib import pyplot as plt

def loadImage(ruta, color=True):
    img = cv2.imread(ruta)
    if (color):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def printImage(img):
    if(img.ndim==2):
        plot_img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        plot_img=img
    plt.imshow(plot_img);
    plt.axis('off');
    plt.show()

def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def crearMascaraGaussiana(MaskSize, sigma):
    sigma2=sigma**2;
    GMask = np.zeros((MaskSize, MaskSize))
    var=np.floor(MaskSize/2).astype('int')
    for i, x in enumerate(range(-var,var+1)):
        for j, y in enumerate(range(-var,var+1)):
            GMask[j,i]=(1/(2*np.pi*sigma2))*(np.e**(-(x*x+y*y)/sigma2))
    GMask=GMask/GMask.sum()
    return GMask

def convolucion(vecindad,Mask):
    var=Mask.shape[0]
    pixel=0
    for i in range(var):
        for j in range(var):
            #print(i,j)
            pixel+=vecindad[i,j]*Mask[i,j]
    return pixel

def aplicarFiltro(img, Mask):
    img_new=np.copy(img)
    var=np.floor(Mask.shape[0]/2).astype('int')
    #Para no tocar los bordes los evito con var
    for y in range(var, img.shape[0]-var):
        for x in range(var, img.shape[1]-var):
            #print('f(%d,%d)' %(x, y))
            vecindad=img[y-var:y+var+1,x-var:x+var+1]
            img_new[y,x] = convolucion(vecindad, Mask).astype('uint8')
    return img_new

def binarizar(img, thr):
    img_new=np.copy(img)
    img_new[img>=thr]=255
    img_new[img<thr]=0
    return img_new