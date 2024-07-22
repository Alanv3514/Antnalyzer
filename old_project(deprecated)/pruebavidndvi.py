import cv2
import numpy as np
from fastiecm import fastiecm

def disp_multiple(im1=None, im2=None, im3=None, im4=None):
    """
    Combines four images for display.
    """
    height, width = im1.shape

    combined = np.zeros((2*height,2*width, 3), dtype=np.uint8)

    combined[0:height, 0:width, :] = cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB)
    combined[height:, 0:width, :] = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)
    combined[0:height, width:, :] = cv2.cvtColor(im3, cv2.COLOR_GRAY2RGB)
    combined[height:, width:, :] = cv2.cvtColor(im4, cv2.COLOR_GRAY2RGB)

    return combined


def contrast_stretch(im):
    """
    Performs a simple contrast stretch of the given image, from 5-95%.
    """
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out

def label(image, text):
    """
    Labels the given image with the given text
    """
    return cv2.putText(image, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

#Prueba de programa de video

cap=cv2.VideoCapture(r"C:\Users\Agustin\Videos\test_3.mp4")
frametime= 16

while True:
    ret, img2= cap.read()
    #roi = img2[100: 350, 200:420] #Region de interes
    b, g, r= cv2.split(img2)

    #Calculamos NDVI
    #Denominador de la fracción

    bottom= (r.astype(float)+b.astype(float))
    bottom[bottom==0] = 0.01

    ndvi=(r.astype(float)-b) / bottom
    ndvi = contrast_stretch(ndvi)
    ndvi = ndvi.astype(np.uint8)

    # Do the labelling
    label(b, 'Blue')
    label(g, 'Green')
    label(r, 'NIR')
    label(ndvi, 'NDVI')

    combined = disp_multiple(b, g, r, ndvi)
    cv2.imshow('image', ndvi)
    color_mapped_prep=ndvi.astype(np.uint8)
    color_mapped_image = cv2.applyColorMap(color_mapped_prep, fastiecm)
    cv2.imshow( 'Color mapped',color_mapped_image)

    if cv2.waitKey(frametime) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()