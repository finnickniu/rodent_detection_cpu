""
import os
import cv2
import sys
import numpy as np
from PIL import Image , ImageDraw
import torch.multiprocessing as mp
from datetime import datetime
from time import gmtime, strftime
from datetime import datetime,time,timedelta
import re
import time
import base64
import argparse
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import datasets, models, transforms

class Server(object):
    def __init__(self,args):
        self.model_path="model.pth"
        self.video_dir = args.video_path
        self.sv_dir = args.ann_dir
        self.device = torch.device(args.device)
        self.score = args.score

    def box2yolo(self,box, image_width, image_height):
        object_class = 0 
        x1,y1,x2,y2 = box
        x = (x1+x2)/2/image_width
        y = (y1+y2)/2/image_height
        w = abs(x1-x2)/image_width
        h = abs(y1-y2)/image_height
        return x,y,w,h



    def model_rcnn(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = 2  # 1 class (person) + background
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load(self.model_path))
        model.to(self.device)
        model.eval()
        return model

    def read_label(self):
        class_map = {"1":"rat"}
        return class_map

    def detection_inference(self,output):
        print(output)
        scores=output["scores"].data
        bb_box=output["boxes"]
        class_=output['labels'].data
        return scores,bb_box,class_

    def write2txt(self,coor,filename,time_period):
        x,y,w,h = coor
        file_name = self.video_dir[:-4] + f"_{time_period}.txt"
        pattern = re.compile(r'@.+')  
        file_name = pattern.findall(file_name)[0]
        file = open(self.sv_dir+file_name, "a") 
        file.write(f"{0} {x} {y} {w} {h}"+"\n") 
        return
    def image2tensor(self,image):
        transform1 = transforms.Compose([
            transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
            ]
        )
        image_pil = Image.fromarray(cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB))
    
        image_tensor=transform1(image_pil)

        image_tensor=image_tensor.unsqueeze(0).to(self.device)
        return image_tensor

        

    @torch.no_grad()
    def run(self):
           
        model_detection = self.model_rcnn()        
        frame =0 
        cap = cv2.VideoCapture(self.video_dir)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cnt = 1 
        while True:
            time_period = cnt/fps*1000
            success,image = cap.read()
            image_height,image_width = image.shape[:2]
            # if not success: 
            #     continue
            image_tensor = self.image2tensor(image)
            labels = self.read_label()
            output = model_detection(image_tensor)[0]

            scores,bb_box,class_ = self.detection_inference(output)
            for i,s in enumerate(scores):
                bbox=bb_box.data[i]
                x1, y1, x2, y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
                if s > self.score:                    
                    cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)
                    x,y,w,h = self.box2yolo([x1, y1, x2, y2], image_width, image_height)
                    self.write2txt([x,y,w,h],self.video_dir,int(time_period))
                   
            cnt+=1
            cv2.imshow('ss1',image)

            fps_time = time.time()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


        
                # else:
                #     sys.exit()
                #     break





    
    
