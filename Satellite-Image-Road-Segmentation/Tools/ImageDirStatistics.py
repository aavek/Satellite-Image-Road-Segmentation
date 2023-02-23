import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

"""
Prints out useful stasitics over images in cropped directories. Thus ignoring (0,0,0) and (255,255,255) pixels when calculating the mean and std.

Example: python ImageDirStatistics.py -d /home/user/images/
"""
#https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
def DominantColor(img, ncolors):
    pixels = np.float32(img.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, ncolors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    return palette[np.argmax(counts)]

def ShowStatistics(infile,dominant):
    try:
        img = cv2.imread(infile)
        (img_mean, img_std) = cv2.meanStdDev(img)
        
        meanList.append(img_mean)
        stdList.append(img_std)
        
        if (dominant):
            img_dom = DominantColor(img,7)
            dominantColorList.append(img_dom)
        
    except IOError:
        print("Cannot get image statistics for ", infile)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", required=True, help = "Directory to look up for images")
    parser.add_argument("-dom", "--dominant", default = False, type=bool, required=False, help = "Whether to find dominant colour")
    args = parser.parse_args()

    input_dir = os.path.normpath(args.d) if args.d else os.getcwd()

    meanList = []
    stdList = []
    dominantColorList = []
    
    for file in tqdm(os.listdir(input_dir)):
        ShowStatistics(os.path.join(input_dir, file),args.dominant)

    print("-"*30)
    print("For images in directory {0}: ".format(input_dir))
    print("Overall Mean : {0}".format(np.mean(meanList,axis=0)))
    print("Overall Standard Deviation : {0}".format(np.mean(stdList,axis=0)))
    if (args.dominant):
        print("Dominant Color in directory: {0}".format(np.mean(dominantColorList,axis=0)))