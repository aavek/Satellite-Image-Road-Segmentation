import collections
import math
import os
import random
import torch
from torch.utils import data
import cv2
import numpy as np
import Tools.sknw as sknw
import Tools.LineSimplification as LineSimp
import Tools.LineConversion as LineConv
import Tools.LineDataExtraction as LineData
from skimage.morphology import skeletonize

class DatasetPreprocessor(data.Dataset):
    def __init__(self, cfg, model_name, dataset_name, loader_type):
        cv2.setNumThreads(0)
        self.cfg = cfg
        self.GraphParameters = [eval(self.cfg["Models"]["scales"]), eval(self.cfg["Models"]["smooth"])]
        np.random.seed(self.cfg["GlobalSeed"])
        torch.manual_seed(self.cfg["GlobalSeed"])
        random.seed(self.cfg["GlobalSeed"])
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.loader_type = loader_type
        self.augment = self.cfg[self.loader_type]["augment"]
        imagelabeldirs = ["train_dir", "train_label_dir"] if self.loader_type == "training_settings" else ["test_dir", "test_label_dir"] # change dirs for test
        pathImages = os.path.abspath(os.curdir + self.cfg["Datasets"][dataset_name][imagelabeldirs[0]])
        pathLabels = os.path.abspath(os.curdir + self.cfg["Datasets"][dataset_name][imagelabeldirs[1]])
        imagelist = os.listdir(pathImages)
        labellist = os.listdir(pathLabels)
        self.image_dir = [os.path.join(pathImages,x) for x in imagelist]
        self.label_dir = [os.path.join(pathLabels,x) for x in labellist]
        assert (len(imagelist) == len(labellist))
        self.dataset_mean_color = np.array(eval(self.cfg["Datasets"][dataset_name]["mean"]))
        self.imagelabeldict = collections.defaultdict(list)
        
        self.crop_size = self.cfg[self.loader_type]["crop_size"]
        if ((self.loader_type == "validation_settings") and (dataset_name == "Spacenet")):
            self.crop_size = self.cfg[self.loader_type]["spacenet_crop_size"]
        
        for image,label in zip(self.image_dir,self.label_dir):
            if ("Moscow" not in image.split("_") and "Mumbai" not in image.split("_")): # Incase Spacenet5.
                self.imagelabeldict[self.loader_type].append({"image" : image, "label" : label})
        
    def __len__(self): 
        return len(self.imagelabeldict[self.loader_type])
        
    def Preprocess(self, index):
        imagelabeldict = self.imagelabeldict[self.loader_type][index]
        image = cv2.imread(imagelabeldict["image"])
        label = cv2.imread(imagelabeldict["label"], 0)
        
        cropsize = self.crop_size
        if self.loader_type == "training_settings":
            image_h_,image_w_,image_c_ = image.shape
            x_crop = np.random.randint(0, image_w_ - cropsize)
            y_crop = np.random.randint(0, image_h_ - cropsize)
            image = image[x_crop : cropsize + x_crop , y_crop : cropsize + y_crop, :]
            label = label[x_crop : cropsize + x_crop , y_crop : cropsize + y_crop]
        elif self.loader_type == "validation_settings":
            image = cv2.resize(image, (cropsize, cropsize), interpolation = cv2.INTER_LINEAR)
            label = cv2.resize(label, (cropsize, cropsize), interpolation = cv2.INTER_LINEAR)
        
        if index == len(self.imagelabeldict[self.loader_type]) - 1:
            np.random.shuffle(self.imagelabeldict[self.loader_type])

        image_h,image_w,image_c = image.shape
        if self.augment:
            flip = np.random.choice(2) * 2 - 1
            image = np.ascontiguousarray(image[:, ::flip, :])
            label = np.ascontiguousarray(label[:, ::flip])
            rot = np.random.randint(4) * 90
            rotM = cv2.getRotationMatrix2D((image_w / 2, image_h / 2), rot, 1)
            image = cv2.warpAffine(image, rotM, (image_w, image_h))
            label = cv2.warpAffine(label, rotM, (image_w, image_h))

        image = image.astype(np.float)
        label = label.astype(np.float)
        
        image -= self.dataset_mean_color

        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(np.array(image))
        
        return image, label
        
    def CalculateAnglesFromVectorMap(self, keypoints, height, width):
        scaled_vector_map, scaled_orientation_angles = LineData.getVectorMapsAngles((height, width), keypoints, theta=10, bin_size=10)
        scaled_orientation_angles = torch.from_numpy(scaled_orientation_angles)
        return scaled_orientation_angles

class DeepGlobe(DatasetPreprocessor):
    def __init__(self, cfg, model_name, dataset_name, loader_type):
        super().__init__(cfg, model_name, "DeepGlobe", loader_type)
        pass
        
    def __getitem__(self, index):
        image, label = self.Preprocess(index)
        scaled_labels = []
        scaled_orient = []
        image_c,image_h,image_w = image.shape
        for i,scale in enumerate(self.GraphParameters[0]):
            scaled_image_h = int(math.ceil(image_h / ((scale * 1.0))))
            scaled_image_w = int(math.ceil(image_w / ((scale * 1.0))))
            if scale != 1:
                scaled_label = cv2.resize(label, (scaled_image_h, scaled_image_w), interpolation = cv2.INTER_NEAREST)
            else:
                scaled_label = label
            scaled_label_ = np.copy(scaled_label)
            scaled_label_ /= 255.0
            scaled_labels.append(scaled_label_)
            
            scaled_skeleton_label = skeletonize(scaled_label_).astype(np.uint16)
            graph_label = sknw.build_sknw(scaled_skeleton_label, multi=True)
            graph_label_road_segments = []
            for (edge_start, edge_end) in graph_label.edges():
                for _, coordinate in graph_label[edge_start][edge_end].items():
                    graph_points = coordinate["pts"]
                    road_segments = np.row_stack([graph_label.nodes[edge_start]["o"], graph_points, graph_label.nodes[edge_end]["o"]])
                    road_segments_simplified = LineSimp.Ramer_Douglas_Peucker(road_segments.tolist(), self.GraphParameters[1][i])
                    graph_label_road_segments.append(road_segments_simplified)
            
            keypoints = LineConv.Graph_to_Keypoints(graph_label_road_segments)
            scaled_orientation_angles = self.CalculateAnglesFromVectorMap(keypoints, height = scaled_image_h, width = scaled_image_w)
            scaled_orient.append(scaled_orientation_angles)
        
        return image, scaled_labels, scaled_orient

class MassachusettsRoads(DatasetPreprocessor):
    def __init__(self, cfg, model_name, dataset_name, loader_type):
        super().__init__(cfg, model_name, "MassachusettsRoads", loader_type)
        pass
        
    def __getitem__(self, index):
        image, label = self.Preprocess(index)
        scaled_labels = []
        scaled_orient = []
        image_c,image_h,image_w = image.shape
        for i,scale in enumerate(self.GraphParameters[0]):
            scaled_image_h = int(math.ceil(image_h / ((scale * 1.0))))
            scaled_image_w = int(math.ceil(image_w / ((scale * 1.0))))
            if scale != 1:
                scaled_label = cv2.resize(label, (scaled_image_h, scaled_image_w), interpolation = cv2.INTER_NEAREST)
            else:
                scaled_label = label
            scaled_label_ = np.copy(scaled_label)
            scaled_label_ /= 255.0
            scaled_labels.append(scaled_label_)
            
            scaled_skeleton_label = skeletonize(scaled_label_).astype(np.uint16)
            graph_label = sknw.build_sknw(scaled_skeleton_label, multi=True)
            graph_label_road_segments = []
            for (edge_start, edge_end) in graph_label.edges():
                for _, coordinate in graph_label[edge_start][edge_end].items():
                    graph_points = coordinate["pts"]
                    road_segments = np.row_stack([graph_label.nodes[edge_start]["o"], graph_points, graph_label.nodes[edge_end]["o"]])
                    road_segments_simplified = LineSimp.Ramer_Douglas_Peucker(road_segments.tolist(), self.GraphParameters[1][i])
                    graph_label_road_segments.append(road_segments_simplified)
            
            keypoints = LineConv.Graph_to_Keypoints(graph_label_road_segments)
            scaled_orientation_angles = self.CalculateAnglesFromVectorMap(keypoints, height = scaled_image_h, width = scaled_image_w)
            scaled_orient.append(scaled_orientation_angles)
        
        return image, scaled_labels, scaled_orient

class Spacenet(DatasetPreprocessor):
    def __init__(self, cfg, model_name, dataset_name, loader_type):
        super().__init__(cfg, model_name, "Spacenet", loader_type)
        self.threshold = self.cfg["training_settings"]["road_binary_thresh"]
        
    def __getitem__(self, index):
        image, label = self.Preprocess(index)
        scaled_labels = []
        scaled_orient = []
        image_c,image_h,image_w = image.shape
        for i,scale in enumerate(self.GraphParameters[0]):
            scaled_image_h = int(math.ceil(image_h / ((scale * 1.0))))
            scaled_image_w = int(math.ceil(image_w / ((scale * 1.0))))
            if scale != 1:
                scaled_label = cv2.resize(label, (scaled_image_h, scaled_image_w), interpolation = cv2.INTER_NEAREST)
            else:
                scaled_label = label
            scaled_label_ = np.copy(scaled_label)
            scaled_label_ /= 255.0
            scaled_labels.append(scaled_label_)
            
            scaled_skeleton_label = skeletonize(scaled_label_).astype(np.uint16)
            graph_label = sknw.build_sknw(scaled_skeleton_label, multi=True)
            graph_label_road_segments = []
            for (edge_start, edge_end) in graph_label.edges():
                for _, coordinate in graph_label[edge_start][edge_end].items():
                    graph_points = coordinate["pts"]
                    road_segments = np.row_stack([graph_label.nodes[edge_start]["o"], graph_points, graph_label.nodes[edge_end]["o"]])
                    road_segments_simplified = LineSimp.Ramer_Douglas_Peucker(road_segments.tolist(), self.GraphParameters[1][i])
                    graph_label_road_segments.append(road_segments_simplified)
            
            keypoints = LineConv.Graph_to_Keypoints(graph_label_road_segments)
            scaled_orientation_angles = self.CalculateAnglesFromVectorMap(keypoints, height = scaled_image_h, width = scaled_image_w)
            scaled_orient.append(scaled_orientation_angles)
            
        return image, scaled_labels, scaled_orient