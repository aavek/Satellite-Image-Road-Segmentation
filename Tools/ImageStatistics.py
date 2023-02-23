import os
import argparse
import cv2
from PIL import Image

"""
Prints out useful image stasitics

Example: python ShowImageStatistics.py -d /home/user/images
"""
def ShowStatistics(infile):
    try:
        img = cv2.imread(infile)
        (img_mean, img_std) = cv2.meanStdDev(img)
    except IOError:
        print("Cannot get image statistics for ", infile)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="Directory to look up for images")
    args = parser.parse_args()

    input_dir = os.path.normpath(args.d) if args.d else os.getcwd()

    for file in os.listdir(input_dir):
        ShowStatistics(os.path.join(input_dir, file))
