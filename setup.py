#!/usr/bin/env python3

"""
    setup.py: This script creates cropped images for each part of a chosen Dataset.
              (DeepGlobe, MassachusettsRoads, Spacenet). Approximate time to create ~= 15min.
              
              The filename format for the cropped images is:
              <filename with no extension>_<tile_row_index>_<tile_column_index>.png
              
              Example (full image cropped into smaller parts):
              |(1,1)|(1,2)|(1,3)|
              |_____|_____|_____|
              |(2,1)|(2,2)|(2,3)|
              |_____|_____|_____|
              |(3,1)|(3,2)|(3,3)| <== This will be <filename>_3_3.png
              |_____|_____|_____|
    
              Directory Structure:
              Script creates corresponding cropped image directories based on dataset structure.
              
              Example (MassachusettsRoads):
              Datasets (existing folder)
              |_______MassachusettsRoads
                      |_______cropped_train
                      |_______cropped_train_labels
                      |_______cropped_valid
                      |_______cropped_valid_labels
                      |_______cropped_test
                      |_______cropped_test_labels
                      ..
                      |_______tiff
                      |_______label_class_dict.csv
                      |_______metadata.csv
    Usage:
    
    python setup.py -d Datasets -cs 512 -j DeepGlobe
    python setup.py -d Datasets -cs 512 -j MassachusettsRoads
    python setup.py -d Datasets -cs 650 -j Spacenet
"""

import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from osgeo import gdal
import random
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
tqdm.monitor_interval = 0
import cv2
from skimage import io
import argparse

def SpacenetContrastEnhancement(image_file,nbands):
    image_banded=np.zeros(image_file.shape,np.uint8)
    for bandId in range(0,nbands):
        image_uint8=np.uint8(image_file[:,:,bandId])
        image_uint8_percentiles = image_uint8[image_uint8 > 0]
        P2=np.percentile(image_uint8_percentiles,2)
        P98=np.percentile(image_uint8_percentiles,98)
        Clipped_Pixels = np.clip(image_uint8, P2, P98)
        image_uint8_clip=np.uint8(((Clipped_Pixels - P2) / (P98 - P2)) * 255)
        image_banded[:,:,bandId]=np.uint8(image_uint8_clip)
    return image_banded

def CroppingProcedure(FileNames,BadImages, DatasetName, FolderPath, CroppedFolderPath, OverlapRatio, CropSize, DataType):
    
    for FileName in tqdm(FileNames, ncols=100, desc="Cropping {0} {1}".format(DatasetName, DataType), total=len(FileNames)):
        FilePath = os.path.join(FolderPath,FileName)
        
        if ("DeepGlobe" in DatasetName):
            if ("mask" not in FileName):
                image = cv2.imread(FilePath)
                if image is None:
                    BadImages.append(FilePath)
                    continue
                CroppingImage(image, FileName, CroppedFolderPath, DatasetName, OverlapRatio, CropSize)
            elif ("mask" in FileName):
                mask = cv2.imread(FilePath)
                if mask is None:
                    BadImages.append(FilePath)
                    continue
                CroppingImage(mask, FileName, CroppedFolderPath, DatasetName, OverlapRatio, CropSize)
                
        elif ("MassachusettsRoads" in DatasetName):
            image = cv2.imread(FilePath)
            if image is None:
                BadImages.append(FilePath)
                continue
            image = cv2.resize(image, dsize=(1536, 1536), interpolation=cv2.INTER_LINEAR)
            if ("labels" not in FolderPath):
                BW = 30
                BorderMask = np.ones(image.shape[:2], dtype = "uint8")
                BorderMaskRect = cv2.rectangle(BorderMask, (BW,BW),(image.shape[1]-BW,image.shape[0]-BW), 0, -1)
                OutputBorder = cv2.bitwise_and(image, image, mask = BorderMaskRect)
                if ((np.count_nonzero(np.all(OutputBorder==[255,255,255],axis=2)) / np.count_nonzero(np.all(OutputBorder!=[0,0,0],axis=2))) > 0.001):
                    BadImages.append(FilePath)
                    continue
                CroppingImage(image, FileName, CroppedFolderPath, DatasetName, OverlapRatio, CropSize)
            elif ("labels" in FolderPath):
                FileNameNoExt = os.path.splitext(FileName)[0]
                if (FileNameNoExt + ".tiff" in [os.path.basename(bFileName) for bFileName in BadImages]):
                    continue
                CroppingImage(image, FileName, CroppedFolderPath, DatasetName, OverlapRatio, CropSize)
                
        elif ("Spacenet" in DatasetName):
            if ("labels" not in FolderPath):
                image_gd = gdal.Open(FilePath)
                if image_gd is None:
                    BadImages.append(FilePath)
                    continue
                nbands = image_gd.RasterCount
                image_gdal=image_gd.ReadAsArray()
                if cv2.countNonZero(image_gdal) == 0:
                    BadImages.append(FilePath)
                    continue
                image = np.transpose(image_gdal,(1,2,0))
                spacenet_image = SpacenetContrastEnhancement(image,nbands)
                image=np.dstack((spacenet_image[:,:,1],spacenet_image[:,:,2],spacenet_image[:,:,4])) # 4,2,1 inverted colors
                BW = 30
                BorderMask = np.ones(image.shape[:2], dtype = "uint8")
                BorderMaskRect = cv2.rectangle(BorderMask, (BW,BW),(image.shape[1]-BW,image.shape[0]-BW), 0, -1)
                OutputBorder = cv2.bitwise_and(image, image, mask = BorderMaskRect)
                nNonZeroBorderPixels = np.count_nonzero(np.all(OutputBorder!=[0,0,0],axis=2))
                if (1 - (nNonZeroBorderPixels / ((BW*image.shape[0])*2 + (BW*image.shape[1])*2 - BW*BW*4)) > 0.25):
                    BadImages.append(FilePath)
                    continue
                
                CroppingImage(image, FileName, CroppedFolderPath, DatasetName, OverlapRatio, CropSize)
                image_gd = None
            elif ("labels" in FolderPath):
                mask = cv2.imread(FilePath)
                FileNameNoExt = os.path.splitext(FileName)[0]
                if ((FileNameNoExt + ".tif" in [os.path.basename(bFileName) for bFileName in BadImages]) or (mask is None)):
                    continue
                CroppingImage(mask, FileName, CroppedFolderPath, DatasetName, OverlapRatio, CropSize)

def CroppingImage(img, FileName, CroppedPath, DatasetName, OverlapRatio, CropSize):
    Rows,Cols,Channels = img.shape
    RowTiles = np.ceil(np.divide(Rows,CropSize)) 
    ColTiles = np.ceil(np.divide(Cols,CropSize))
    RowTilesBetween = RowTiles - (OverlapRatio - 1)
    ColTilesBetween = ColTiles - (OverlapRatio - 1)
    if (OverlapRatio !=1):
        RowTiles += RowTilesBetween
        ColTiles += ColTilesBetween
    RowStride=int((CropSize*RowTiles - Rows) / (RowTiles-1))
    ColStride=int((CropSize*ColTiles - Cols) / (ColTiles-1)) 
    
    ColIdx=0
    for Col in range (0, Cols - CropSize + 2, (CropSize - ColStride)):
        ColIdx += 1
        if (Col + CropSize > Cols):
            Col=Col-1
        RowIdx=0
        for Row in range (0,Rows - CropSize + 2, (CropSize - RowStride)):
            RowIdx += 1
            if (Row+CropSize > Rows):
                Row=Row-1
            cropped_img = img[Row:Row+CropSize, Col:Col+CropSize,:]
            FileNameNoExtension = os.path.splitext(FileName)[0]
            cv2.imwrite(os.path.join(CroppedPath,"{}_{}_{}.png".format(FileNameNoExtension,RowIdx,ColIdx)), cropped_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])

def PrepareDatasetsForProcessing(DataPath, CropSize, JustOne):
    
    for i,DatasetName in enumerate(os.listdir(DataPath)):
    
        if ((JustOne is not None) and (DatasetName != JustOne)):
            continue
            
        print("-"*30)
        DatasetPath = os.path.join(os.getcwd(),DataPath,DatasetName)
        print(DatasetPath)
        print("Working on dataset: {0}".format(DatasetName))
        
        CroppedTrainValPaths = [os.path.join(DatasetPath,"cropped_train"),
                                os.path.join(DatasetPath,"cropped_train_labels"),
                                os.path.join(DatasetPath,"cropped_valid"),
                                os.path.join(DatasetPath,"cropped_valid_labels")]
        
        for CroppedTrainValPath in CroppedTrainValPaths:
            if not os.path.exists(CroppedTrainValPath):
                os.mkdir(CroppedTrainValPath)
        
        if os.path.exists(DatasetPath) and "DeepGlobe" in DatasetName: # https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset
            TrainPath = os.path.join(DatasetPath,"train")
            TestPath = os.path.join(DatasetPath,"valid")
            HoldPath = os.path.join(DatasetPath,"test")
            
            TrainValPathFiles = os.listdir(TrainPath)
            TrainValFiles = [x for i,x in enumerate(TrainValPathFiles) if os.path.splitext(TrainValPathFiles[i])[0][-4:] == "_sat"]
            TrainValMaskFiles = [x for i,x in enumerate(TrainValPathFiles) if os.path.splitext(TrainValPathFiles[i])[0][-4:] == "mask"]
            TrainFiles, ValFiles, TrainMaskFiles, ValMaskFiles = train_test_split(TrainValFiles, TrainValMaskFiles, test_size = int(1530), random_state=7) # For test_size: If int, represents the absolute number of test samples
            
            TestFiles = os.listdir(TestPath)
            HoldFiles = os.listdir(HoldPath)
            
            CroppedTestPath = os.path.join(DatasetPath,"cropped_test")
            CroppedHoldPath = os.path.join(DatasetPath,"cropped_hold")
                
            if not os.path.exists(CroppedTestPath):
                os.mkdir(CroppedTestPath)
            if not os.path.exists(CroppedHoldPath):
                os.mkdir(CroppedHoldPath)
            
            BadImages = []
            CroppingProcedure(TrainFiles, BadImages, DatasetName, TrainPath, CroppedTrainValPaths[0], 2, CropSize, "Training Images")
            CroppingProcedure(TrainMaskFiles, BadImages, DatasetName, TrainPath, CroppedTrainValPaths[1], 2, CropSize, "Training Masks")
            CroppingProcedure(ValFiles, BadImages, DatasetName, TrainPath, CroppedTrainValPaths[2], 1, CropSize, "Validation Images")
            CroppingProcedure(ValMaskFiles, BadImages, DatasetName, TrainPath, CroppedTrainValPaths[3], 1, CropSize, "Validation Masks")
            CroppingProcedure(TestFiles, BadImages, DatasetName, TestPath, CroppedTestPath, 1, CropSize, "Testing Images")
            CroppingProcedure(HoldFiles, BadImages, DatasetName, HoldPath, CroppedHoldPath, 1, CropSize, "HoldOut Images")
            if len(BadImages) > 0:
                print("{0} Bad {1} Images : {2}".format(len(BadImages), DatasetName, BadImages))
        
        elif os.path.exists(DatasetPath) and "MassachusettsRoads" in DatasetName: # https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset
            TrainPath = os.path.join(DatasetPath,"tiff","train")
            TrainMaskPath = os.path.join(DatasetPath,"tiff","train_labels")
            ValidPath = os.path.join(DatasetPath,"tiff","val")
            ValidMaskPath = os.path.join(DatasetPath,"tiff","val_labels")
            TestPath = os.path.join(DatasetPath,"tiff","test")
            TestMaskPath = os.path.join(DatasetPath,"tiff","test_labels")
            
            TrainFiles = os.listdir(TrainPath)
            TrainMaskFiles = os.listdir(TrainMaskPath)
            ValFiles = os.listdir(ValidPath)
            ValMaskFiles = os.listdir(ValidMaskPath)
            TestFiles = os.listdir(TestPath)
            TestMaskFiles = os.listdir(TestMaskPath)
            
            CroppedTestPath = os.path.join(DatasetPath,"cropped_test")
            CroppedTestMaskPath = os.path.join(DatasetPath,"cropped_test_labels")
            if not os.path.exists(CroppedTestPath):
                os.mkdir(CroppedTestPath)
            if not os.path.exists(CroppedTestMaskPath):
                os.mkdir(CroppedTestMaskPath)
            
            BadImages = []
            CroppingProcedure(TrainFiles, BadImages, DatasetName, TrainPath, CroppedTrainValPaths[0], 2, CropSize, "Training Images")
            CroppingProcedure(TrainMaskFiles, BadImages, DatasetName, TrainMaskPath, CroppedTrainValPaths[1], 2, CropSize, "Training Masks")
            CroppingProcedure(ValFiles, BadImages, DatasetName, ValidPath, CroppedTrainValPaths[2], 1, CropSize, "Validation Images")
            CroppingProcedure(ValMaskFiles, BadImages, DatasetName, ValidMaskPath, CroppedTrainValPaths[3], 1, CropSize, "Validation Masks")
            CroppingProcedure(TestFiles, BadImages, DatasetName, TestPath, CroppedTestPath, 1, CropSize, "Testing Images")
            CroppingProcedure(TestMaskFiles, BadImages, DatasetName, TestMaskPath, CroppedTestMaskPath, 1, CropSize, "Testing Masks")
            if len(BadImages) > 0:
                print("{0} Bad {1} Images : {2}".format(len(BadImages), DatasetName, BadImages))
              
        elif os.path.exists(DatasetPath) and "Spacenet" in DatasetName:
            TrainValPath = os.path.join(DatasetPath,"trainval")
            TrainValMaskPath = os.path.join(DatasetPath,"trainval_labels","train_masks")
            TestPath = os.path.join(DatasetPath,"test")
            
            TrainValFiles = os.listdir(TrainValPath)
            TrainValMaskFiles  = os.listdir(TrainValMaskPath)
            TrainFiles, ValFiles, TrainMaskFiles, ValMaskFiles = train_test_split(TrainValFiles, TrainValMaskFiles, test_size = int(1050), random_state=7)
            
            TestFiles = os.listdir(TestPath)
            CroppedTestPath = os.path.join(DatasetPath,"cropped_test")
            if not os.path.exists(CroppedTestPath):
                os.mkdir(CroppedTestPath)
            
            BadImages = []
            CropSize = 650
            CroppingProcedure(TrainFiles, BadImages, DatasetName, TrainValPath, CroppedTrainValPaths[0], 2, CropSize, "Training Images")
            CroppingProcedure(TrainMaskFiles, BadImages, DatasetName, TrainValMaskPath, CroppedTrainValPaths[1], 2, CropSize, "Training Masks")
            CroppingProcedure(ValFiles, BadImages, DatasetName, TrainValPath, CroppedTrainValPaths[2], 1, CropSize, "Validation Images")
            CroppingProcedure(ValMaskFiles, BadImages, DatasetName, TrainValMaskPath, CroppedTrainValPaths[3], 1, CropSize, "Validation Masks")
            CroppingProcedure(TestFiles, BadImages, DatasetName, TestPath, CroppedTestPath, 1, CropSize, "Testing Images")
            if len(BadImages) > 0:
                print("{0} Bad {1} Images : {2}".format(len(BadImages), DatasetName, BadImages))
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--DataPath", type=str, required=True, help = "Path to Datasets")
    parser.add_argument("-cs", "--CropSize", type=int, required=True, help = "Size of Cropping Window")
    parser.add_argument("-j", "--JustOne", default=None, type=str, required=False, help = "Perform Setup on just one Dataset: DeepGlobe, MassachusettsRoads, or Spacenet")
    args = parser.parse_args()
    
    if os.listdir(args.DataPath) != []:
        print("-"*30)
        print("Found the following datasets: ")
        for i,folders in enumerate(os.listdir(args.DataPath)):
            print('{0}) {1}'.format(i+1,folders))
    nfolders = len(os.listdir(args.DataPath))
    
    dataset_time_start = time.perf_counter()
    PrepareDatasetsForProcessing(args.DataPath, args.CropSize, args.JustOne)
    dataset_time_end = time.perf_counter()
    print("Prepared {0} datasets in {1:.2g} seconds".format(nfolders,(dataset_time_end - dataset_time_start)))
    
if __name__=="__main__":
    main()