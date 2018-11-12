import os, random
import cv2

def random_rotation(img):

    rows = img.shape[0]
    cols = img.shape[1]
    angle = random.randint(0, 360)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img = cv2.warpAffine(img, M, (cols, rows))

    return img

def get_random_mnist(path):
    digit = random.randint(0, 9)
    if digit < 0 or digit > 9:
        exit(1)

    chosen = random.choice(os.listdir(path + '/'+ str(digit)))
    return (path + '/'+ str(digit) + "/" + chosen), digit