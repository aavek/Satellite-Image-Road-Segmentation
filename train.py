import os
import sys
import math
import time
import json
import numpy as np
from osgeo import gdal
import random
import argparse
from Models import ConvNeXt_UPerNet_DGCN_MTL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm
from Tools import DatasetUtility
from Tools import Losses
from Tools import util
from Tools import viz_util
tqdm.monitor_interval = 0
import cv2
from skimage import io
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_model(model, nGPUs, cfg, train_loss_file, valid_loss_file, train_loss_angle_file, valid_loss_angle_file, 
                train_loader, valid_loader, Optimizer, segmentation_loss, orientation_loss, 
                LR_scheduler, Epoch, resume_at_epoch, n_RoadClasses, n_OrientClasses):
    train_loss_road = 0
    train_loss_angle = 0
    model.train()
    hist = np.zeros((n_RoadClasses, n_RoadClasses))
    hist_angles = np.zeros((n_OrientClasses, n_OrientClasses))
    crop_size = cfg["training_settings"]["crop_size"]
    i_freq = cfg["training_settings"]["iteration_frequency"]
    for i, ImageLabelData in enumerate(train_loader, 0):
        imageBGRcropped, scaled_target_road_label, scaled_target_orientation_class = ImageLabelData
        
        imageBGRcropped = imageBGRcropped.float().cuda()
        scaled_target_road_label = [_label.cuda() for _label in scaled_target_road_label] 
        scaled_target_orientation_class = [_label.cuda() for _label in scaled_target_orientation_class]
        
        predictions = model(imageBGRcropped)
        predicted_road = [x for x in predictions[0]]
        predicted_orientation_class = [x for x in predictions[1]]
        
        road_loss = segmentation_loss(predicted_road[0], scaled_target_road_label[0])
        for r in range(1,len(predicted_road)):
            road_loss += segmentation_loss(predicted_road[r], scaled_target_road_label[r])
            
        angle_loss = orientation_loss(predicted_orientation_class[0], scaled_target_orientation_class[0])
        for r in range(1,len(predicted_orientation_class)):
            angle_loss += orientation_loss(predicted_orientation_class[r], scaled_target_orientation_class[r])

        train_loss_road += road_loss.item()
        train_loss_angle += angle_loss.item()

        predicted_road = predicted_road[-1]
        predicted_orientation_class = predicted_orientation_class[-1]
        
        _, predicted_road_ = torch.max(predicted_road, 1)
        _, predicted_angle_ = torch.max(predicted_orientation_class, 1)

        target_road = scaled_target_road_label[-1].view(-1, crop_size, crop_size).long()
        target_angle = scaled_target_orientation_class[-1].view(-1, crop_size, crop_size).long()
        
        hist += util.fast_hist(predicted_road_.view(predicted_road_.size(0), -1).cpu().numpy(),
                               target_road.view(target_road.size(0), -1).cpu().numpy(), n_RoadClasses)
                               
        hist_angles += util.fast_hist(predicted_angle_.view(predicted_angle_.size(0), -1).cpu().numpy(),
                                      target_angle.view(target_angle.size(0), -1).cpu().numpy(),n_OrientClasses)
                                      
        p_accu, miou, road_iou, fwacc = util.performMetrics(train_loss_file, valid_loss_file, 
                                                            Epoch, hist, 
                                                            train_loss_road / (i + 1), train_loss_angle / (i + 1), is_train = True, write=True)
                                                            
        p_accu_angle, miou_angle, fwacc_angle = util.performAngleMetrics(train_loss_angle_file, valid_loss_angle_file, 
                                                                         Epoch, hist_angles, is_train = True, write=True)

        viz_util.progress_bar(i, len(train_loader),
            "Loss: %.6f | VecLoss: %.6f | road miou: %.4f%%(%.4f%%) | angle miou: %.4f%% "
            % (train_loss_road / (i + 1), train_loss_angle / (i + 1), miou, road_iou, miou_angle))

        Optimizer.zero_grad()
        torch.autograd.backward([road_loss, angle_loss])

        if (i % i_freq == 0) or (i == len(train_loader) - 1):
            Optimizer.step()
            
        del (predicted_road,
            predicted_orientation_class,
            predicted_road_, predicted_angle_,
            target_road, target_angle,
            imageBGRcropped,
            scaled_target_road_label,
            scaled_target_orientation_class)

    util.performMetrics(train_loss_file, valid_loss_file,
                        Epoch, hist,
                        train_loss_road / len(train_loader),
                        train_loss_angle / len(train_loader),
                        write=True)
                        
    util.performAngleMetrics(train_loss_angle_file, valid_loss_angle_file, Epoch, hist_angles, write=True)
        
def validate_model(ExperimentDirectory, model, dataset_name, nGPUs, cfg, train_loss_file, valid_loss_file, train_loss_angle_file, valid_loss_angle_file, 
                   train_loader, valid_loader, Optimizer, segmentation_loss, orientation_loss, 
                   LR_scheduler, Epoch, n_RoadClasses, n_OrientClasses):
    global best_accuracy
    global best_miou
    model.eval()
    valid_loss_road = 0
    valid_loss_angle = 0
    hist = np.zeros((n_RoadClasses, n_RoadClasses))
    hist_angles = np.zeros((n_OrientClasses, n_OrientClasses))
    
    crop_size = cfg["validation_settings"]["crop_size"]
    if dataset_name == "Spacenet":
       crop_size = cfg["validation_settings"]["spacenet_crop_size"]
    
    with torch.no_grad():
        for i, ImageLabelData in enumerate(valid_loader, 0):
            imageBGR, scaled_target_road_label, scaled_target_orientation_class = ImageLabelData

            imageBGR = imageBGR.float().cuda()
            imageBGR.requires_grad = False
            
            scaled_target_road_label = [_label.cuda() for _label in scaled_target_road_label]
            scaled_target_orientation_class = [_label.cuda() for _label in scaled_target_orientation_class]
            
            predictions = model(imageBGR)
            predicted_road = [x for x in predictions[0]]
            predicted_orientation_class = [x for x in predictions[1]]
            
            if dataset_name == "Spacenet":
                for ipr,resize_predicted_road in enumerate(predicted_road):
                    predicted_road[ipr] = F.interpolate(resize_predicted_road, size=scaled_target_road_label[ipr].shape[-2:], mode='bilinear', align_corners=False)
                for ipoc,resize_predicted_orientation_class in enumerate(predicted_orientation_class):
                    predicted_orientation_class[ipoc] = F.interpolate(resize_predicted_orientation_class, size=scaled_target_orientation_class[ipoc].shape[-2:], mode='bilinear', align_corners=False)
            
            road_loss = segmentation_loss(predicted_road[0], scaled_target_road_label[0])
            for r in range(1,len(predicted_road)):
                road_loss += segmentation_loss(predicted_road[r], scaled_target_road_label[r])
                
            angle_loss = orientation_loss(predicted_orientation_class[0], scaled_target_orientation_class[0])
            for r in range(1,len(predicted_orientation_class)):
                angle_loss += orientation_loss(predicted_orientation_class[r], scaled_target_orientation_class[r])
                
            valid_loss_road += road_loss.item()
            valid_loss_angle += angle_loss.item()

            predicted_road = predicted_road[-1]
            predicted_orientation_class = predicted_orientation_class[-1]
            
            _, predicted_road_ = torch.max(predicted_road, 1)
            _, predicted_angle_ = torch.max(predicted_orientation_class, 1)
            
            target_road = scaled_target_road_label[-1].view(-1, crop_size, crop_size).long()
            target_angle = scaled_target_orientation_class[-1].view(-1, crop_size, crop_size).long()
            
            hist += util.fast_hist(predicted_road_.view(predicted_road_.size(0), -1).cpu().numpy(),
                                   target_road.view(target_road.size(0), -1).cpu().numpy(), n_RoadClasses)
                                   
            hist_angles += util.fast_hist(predicted_angle_.view(predicted_angle_.size(0), -1).cpu().numpy(),
                                          target_angle.view(target_angle.size(0), -1).cpu().numpy(),n_OrientClasses)
            
            p_accu, miou, road_iou, fwacc = util.performMetrics(train_loss_file, valid_loss_file, 
                                                                Epoch, hist, 
                                                                valid_loss_road / (i + 1), valid_loss_angle / (i + 1), is_train=False, write=True)
                                                                
            p_accu_angle, miou_angle, fwacc_angle = util.performAngleMetrics(train_loss_angle_file, valid_loss_angle_file, 
                                                                             Epoch, hist_angles, is_train=False, write=True)
            
            viz_util.progress_bar(i, len(valid_loader),
                "Loss: %.6f | VecLoss: %.6f | road miou: %.4f%%(%.4f%%) | angle miou: %.4f%% "
                % (valid_loss_road / (i + 1), valid_loss_angle / (i + 1), miou, road_iou, miou_angle))
            

            if i % 10 == 0 or i == len(valid_loader) - 1:
                images_path = "{}/images/".format(ExperimentDirectory)
                util.ensure_dir(images_path)
                util.savePredictedProb(
                    imageBGR.data.cpu(),
                    scaled_target_road_label[-1].cpu(),
                    predicted_road_.cpu(),
                    F.softmax(predicted_road, dim=1).data.cpu()[:, 1, :, :],
                    predicted_angle_.cpu(),
                    os.path.join(images_path, "validate_pair_{}_{}.png".format(Epoch, i)),
                    norm_type="Mean")

            del (predicted_road,
                predicted_orientation_class,
                predicted_road_, predicted_angle_,
                target_road, target_angle,
                imageBGR,
                scaled_target_road_label,
                scaled_target_orientation_class)

    accuracy, miou, road_iou, fwacc = util.performMetrics(train_loss_file, valid_loss_file,
                                                          Epoch, hist,
                                                          valid_loss_road / len(valid_loader), valid_loss_angle / len(valid_loader), 
                                                          is_train=False, write=True)
    util.performAngleMetrics(train_loss_angle_file, valid_loss_angle_file,
                             Epoch, hist_angles, is_train=False, write=True)

    if miou > best_miou:
        best_accuracy = accuracy
        best_miou = miou
        util.save_checkpoint(Epoch, valid_loss_road / len(valid_loader), model, Optimizer, best_accuracy, best_miou, cfg, ExperimentDirectory)

    return valid_loss_road / len(valid_loader)

def main():

    nGPUs = torch.cuda.device_count()
    with open("cfg.json", 'r') as f:
        cfg = json.load(f)
    Seed = cfg["GlobalSeed"]
    Epochs = cfg["training_settings"]["epochs"]\
    
    # Models
    ModelFolderList = os.listdir(cfg["Models"]["base_dir"])
    ModelNames = [os.path.splitext(x)[0] for x in ModelFolderList if ((os.path.splitext(x)[1] == ".py"))]
    ModelNames.sort(key=str.lower)
    ModuleNames = ["ConvNeXt_UPerNet_DGCN_MTL"]
    ModuleNames.sort(key=str.lower)
    ModelModuleNames = list(map('.'.join, zip(ModelNames, ModuleNames)))
    ModuleList = [eval(x) for x in ModelModuleNames]
    ChosenModel = dict(zip(ModelNames, ModuleList))
    
    # Datasets
    DatasetNames = os.listdir(cfg["Datasets"]["base_dir"])
    DatasetClassNames = list(map('.'.join, zip(["DatasetUtility"]*len(DatasetNames), DatasetNames)))
    DatasetList = [eval(y) for y in DatasetClassNames]
    ChosenDataset = dict(zip(DatasetNames, DatasetList))
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default='all', const='all', type=str, nargs='?', choices = ModelNames, help = ModelNames)
    parser.add_argument("-d", "--dataset", default='all', const='all', type=str, nargs='?', choices = DatasetNames, help = DatasetNames)
    parser.add_argument("-e", "--experiment", required=True, type=str, help = "Experiment Name")
    parser.add_argument("-r", "--resume", required=False, type=str, default=None, help="Most recent checkpoint (.pt) location (default: None)")
    parser.add_argument("-rd", "--resumedataset", required=False, type=str, default=None, help="Most recent checkpoint (.pt) location (default: None)")
    args = parser.parse_args()
    
    ExperimentDirectory = os.path.join(cfg["training_settings"]["results_directory"], args.experiment)
    if not os.path.exists(ExperimentDirectory):
        os.makedirs(ExperimentDirectory)
    
    model = ChosenModel[args.model]()
    dataset = ChosenDataset[args.dataset]
    
    if nGPUs == 0:
        print("torch.cuda can not find any GPUs on this device. Aborting...")
        sys.exit()
    elif nGPUs == 1:
        model.cuda()
    else:
        model = nn.DataParallel(model).cuda()
    
    Optimizer = optim.SGD(model.parameters(), 
                              lr = cfg["optimizer_settings"]["learning_rate"], 
                              momentum=0.9, 
                              weight_decay = cfg["optimizer_settings"]["learning_rate_decay"])

    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["state_dict"])
        Optimizer.load_state_dict(checkpoint["optimizer"])
        resume_at_epoch = checkpoint["epoch"] + 1
        epoch_with_best_miou = checkpoint["miou"]
    elif args.resumedataset is not None:
        checkpoint = torch.load(args.resumedataset)
        model.load_state_dict(checkpoint["state_dict"])
        Optimizer.load_state_dict(checkpoint["optimizer"])
        resume_at_epoch = 1
        epoch_with_best_miou = checkpoint["miou"]
    else:
        resume_at_epoch = 1
        np.random.seed(Seed)
        torch.manual_seed(Seed)
        torch.cuda.manual_seed_all(Seed)
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                v = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight.data, 0.0, math.sqrt(2.0 / v))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
                
    LR_scheduler = MultiStepLR(Optimizer, 
                               milestones = eval(cfg["optimizer_settings"]["learning_rate_drop_at_epoch"]), 
                               gamma = cfg["optimizer_settings"]["learning_rate_step"])
    
    train_loader = data.DataLoader(dataset(cfg, args.model, args.dataset, "training_settings"), 
                                   batch_size = cfg["training_settings"]["batch_size"], 
                                   num_workers=4, 
                                   shuffle=True, 
                                   pin_memory=False)
    
    valid_loader = data.DataLoader(dataset(cfg, args.model, args.dataset, "validation_settings"), 
                                   batch_size = cfg["validation_settings"]["batch_size"], 
                                   num_workers=4, 
                                   shuffle=False, 
                                   pin_memory=False)
                                   
    n_RoadClasses = cfg["training_settings"]["roadclass"]
    n_OrientClasses = cfg["training_settings"]["orientationclass"]
    segmentation_weights = torch.ones(n_RoadClasses).cuda()
    orientation_weights = torch.ones(n_OrientClasses).cuda()

    segmentation_loss = Losses.mIoULoss(weight = segmentation_weights, n_classes = n_RoadClasses).cuda()
    orientation_loss = Losses.CrossEntropyLossImage(weight=orientation_weights, ignore_index=255).cuda() 
    
    train_file = "{}/{}_train_loss.txt".format(ExperimentDirectory, args.dataset)
    valid_file = "{}/{}_valid_loss.txt".format(ExperimentDirectory, args.dataset)
    train_loss_file = open(train_file, "w")
    valid_loss_file = open(valid_file, "w")
    
    train_file_angle = "{}/{}_train_angle_loss.txt".format(ExperimentDirectory, args.dataset)
    valid_file_angle = "{}/{}_valid_angle_loss.txt".format(ExperimentDirectory, args.dataset)
    train_loss_angle_file = open(train_file_angle, "w")
    valid_loss_angle_file = open(valid_file_angle, "w")
    
    for Epoch in range(resume_at_epoch, Epochs + 1):
        Epoch_Start_Time = time.perf_counter()
        print("\nTraining Epoch: %d" % Epoch)
        train_model(model, nGPUs, cfg, train_loss_file, valid_loss_file, train_loss_angle_file, valid_loss_angle_file, 
                    train_loader, valid_loader, Optimizer, segmentation_loss, orientation_loss, 
                    LR_scheduler, Epoch, resume_at_epoch, n_RoadClasses, n_OrientClasses)
        LR_scheduler.step()
        if (Epoch % cfg["validation_settings"]["evaluation_frequency"] == 0) or (Epoch>90):
            print("\nTesting Epoch: %d" % Epoch)
            val_loss = validate_model(ExperimentDirectory, model, args.dataset, nGPUs, cfg, train_loss_file, valid_loss_file, train_loss_angle_file, valid_loss_angle_file, 
                                      train_loader, valid_loader, Optimizer, segmentation_loss, orientation_loss, 
                                      LR_scheduler, Epoch, n_RoadClasses, n_OrientClasses)
            
        Epoch_End_Time = time.perf_counter()
        print("Time Elapsed for Epoch : {1}".format(Epoch, Epoch_End_Time - Epoch_Start_Time))
    
if __name__=="__main__":
    best_accuracy = 0
    best_miou = 0
    main()