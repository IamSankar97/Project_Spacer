import cv2
import numpy as np
import os.path
import multiprocessing
from itertools import starmap

NBHD_SIZE = 19
UNSHARP_T = 48
ADAPT_T = 24

INVERT = True
ASPECT = 8.5 / 11.0


class PHOTOCOPY:
    def __init__(self, NBHD_SIZE_=19):
        self.NBHD_SIZE = NBHD_SIZE_

    def __call__(self, img_, NBHD_SIZ=19):
        img = img_.copy()
        img = 255 - img
        return self.bitone(img, NBHD_SIZ)

    def bitone(self, image, NBHD_SIZE):
        '''
        Convert a greyscale image to a bitone image,
        in such a way that we preserve as much detail as possible,
        and have the least amount of speckles.
        '''
        # First, sharpen the image: unsharp mask w/ threshold.
        blur = cv2.blur(image, (NBHD_SIZE, NBHD_SIZE))
        diff = cv2.absdiff(image, blur)

        return diff


def convert_to_gray(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def unevenLightCompensate(img, blockSize):
    img = convert_to_gray(img)
    average = np.mean(img)

    rows_new = int(np.ceil(img.shape[0] / blockSize))
    cols_new = int(np.ceil(img.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > img.shape[0]):
                rowmax = img.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > img.shape[1]):
                colmax = img.shape[1]

            imageROI = img[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    img2 = img.astype(np.float32)
    dst = img2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (3, 3), 0)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    return dst


def remove_back_ground(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # h, w = gray_img.shape[:2]
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    _, binary = cv2.threshold(blur_img, 30, 255, cv2.THRESH_BINARY)
    RM_BG_img = cv2.bitwise_and(gray_img, gray_img, mask=binary)
    return RM_BG_img

    # else:
    #     return blur_img
