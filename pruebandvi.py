import cv2
import numpy as np

original = cv2.imread(r"C:\Users\Agustin\Documents\Python\opencv\assets\hojas1.jpg")


def display(image, image_name):
    image = np.array(image, dtype=float)/float(255)
    shape = image.shape
    height = int(shape[0] / 2)
    width = int(shape[1] / 2)
    image = cv2.resize(image, (width, height))
    cv2.namedWindow(image_name)
    cv2.imshow(image_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def contrast_stretch(im):
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)
    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_max - out_min) / (in_max - in_min))
    out += in_min

    return out

def calc_ndvi(image):
    b, g, r = cv2.split(image)
    bottom = (r.astype(float) + b.astype(float))
    bottom[bottom==0] = 0.01
    ndvi = (b.astype(float) - r) / bottom
    return ndvi

display(original, 'Original')
contrasted = contrast_stretch(original)
display(contrasted, 'Contrasted original')
ndvi = calc_ndvi(contrasted)
contrasted_ndvi=contrast_stretch(ndvi)

display(contrasted_ndvi, 'NDVI')