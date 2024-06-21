
#%%
import torch
from torchvision.datasets import DatasetFolder
import ast
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import numpy as np
from torch_snippets import Report
import os
import torch
import cv2
import pandas as pd
from typing import List, Union
from PIL import Image

train_label_path = "/home/lin/codebase/hse_count_prediction/zindi_hse_no_pred/Train.csv"

train_df = pd.read_csv(train_label_path)


label2target = {l:t+1 for t,l in enumerate(train_df["category_id"].unique())}
label2target["background"] = 0
target2label = {t:l for l,t in label2target.items()}
background_class = label2target["background"]

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"

category_map = {1: "Other", 2: "Tin", 3: "Thatch"}


def preprocess_img(img, device=device):
    img = torch.tensor(img).permute(2,0,1)
    return img.float()
    
class HseDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, only_bbox, preprocess_fun=preprocess_img,
                 return_image_path=False,
                 resize=(224,224)
                 ):
        self.df = df
        self.img_dir = img_dir
        self.preprocess_fun = preprocess_fun
        self.img_info = self.df.image_id.unique()
        self.only_bbox = only_bbox
        self.return_image_path = return_image_path
        self.resize = resize
        self.files = glob(f"{self.img_dir}/*")
        
        #if self.only_bbox:
            #self.label2target = {l:t+1 for t,l in enumerate(self.df["image_id"].unique())}
        self.label2target = {"Other": 0, "Tin": 1, "Thatch": 2}
        #else:
            #self.label2target = {"background": 0, "Other": 1.0, "Tin": 2.0, "Thatch": 3.0}
            #self.label2target = {l:t+1 for t,l in enumerate(self.df["image_id"].unique())}
        #self.label2target["background"] = 0
        self.target2label = {t:l for l,t in self.label2target.items()}
        #self.background_class = self.label2target["background"]
            
        
    def __getitem__(self, index):
        image_id = self.img_info[index]
        img_file_path = [img_p for img_p in self.files if
                        os.path.basename(img_p).split(".")[0] == image_id
                        ][0]
        #print(f"img_file_path: {img_file_path}")
        #img_name = os.path.basename(img_file_path).split(".")[0]
        img_df = self.df[self.df["image_id"]==image_id]
        img_bbox = img_df["bbox"].values .tolist()#[0]
        img_label = img_df["category_id"].values.tolist()
        #print(img_bbox)
        bbox = []
        for bbx in img_bbox:
            if isinstance(bbx, str):
                bb = ast.literal_eval(bbx)
                xmin,ymin, w, h = bb
                xmax = xmin + w
                ymax = ymin + h
                if (xmin == xmax) or (ymin == ymax):
                    print(f"found issue with {img_file_path}")
                    print(f"bbox: {xmin, ymin, xmax, ymax}")
                    pass
                else:
                    bbox.append([xmin, ymin, xmax, ymax])
                #category_id = df_ind.category_id
            else:
                bbox = None
        img_bbox = bbox  
        #print(bbox)          
        #print(img_bbox)
        img = cv2.imread(img_file_path)
        self.orig_height, self.orig_width = img.shape[0], img.shape[1]
        if self.resize:
            #print(f"RUNNING RESIZING: {self.resize}")
            # img_bbox = (img_df["xmin"].values[0], img_df["ymin"].values[0],
            #             img_df["xmax"].values[0], img_df["ymax"].values[0]
            #             )
            #print(f"img_bbox: {img_bbox}")        
            
            resize_height, resize_width = self.resize[0], self.resize[1]
            img = np.resize(img, (resize_height, resize_width))
            ratio_height = resize_height / self.orig_height
            ratio_width = resize_width / self.orig_width
            #resize_bbx_xmin, resize_bbx_xmax = img_bbox[0] * ratio_width, img_bbox[2] * ratio_width
            #resize_bbx_ymin, resize_bbx_ymax = img_bbox[1] * ratio_height, img_bbox[3] * ratio_height
            # resized_bbox = [resize_bbx_xmin, resize_bbx_ymin,
            #                 resize_bbx_xmax, resize_bbx_ymax
            #                 ]
            # img_bbox = np.array(resized_bbox)
            # print(f"resized_bbox: {img_bbox}")
            img_bbox = [np.float64(bbx) for bbx in bbox]
            #img_bbox[:, [0,2]] = img_bbox[:, [0,2]].values * ratio_width
            #img_bbox[: [1,3]] = img_bbox[: [1,3]].values * ratio_height.astype(np.float16())
            _img_bbox = []
            for bbx in img_bbox:
                bbx[[0,2]] *= ratio_width
                bbx[[1,3]] *= ratio_height
                _img_bbox.append([np.uint32(b) for b in bbx])
            img_bbox = _img_bbox #np.int32(_img_bbox)
            
            for be in img_bbox:
                if be == [710.7142944335938, 582.142822265625, 714.2857055664062, 582.142822265625]:
                    print(f"found in {img_file_path}")
                    print(be)
                    exit()
            #print(f"img_bbox: {img_bbox}")
        #print(f"img: {img}")    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)/255
        target = {}
        #img_bbox_uint32 = img_bbox.astype(np.uint32).tolist()
        #img_bbox_uint32 = [bbx.astype(np.uint32) for bbx in img_bbox]
        #img_bbox_uint32 = [torch.Tensor(img_bb_unit32).float().unsqueeze(0) for img_bb_unit32 in img_bbox_uint32]
        #print(f"img_bbox_uint32: {img_bbox_uint32}")
        
        if self.only_bbox:
            #img_label = img_df["category_id"].values.tolist()
            target["boxes"] = torch.Tensor(img_bbox).float()#.squeeze(0) #img_bbox_uint32]) # torch.Tensor(img_bbox_uint32).float().unsqueeze(0)
            #target["labels"] = torch.Tensor([self.label2target[i] for i in img_label]).long()
            target["labels"] = torch.Tensor(img_label).long()#.squeeze(0)
            #print(f"target['boxes']: {target['boxes']}")
            #print(f"target['labels']: {target['labels']}")
        else:
            target["boxes"] = torch.Tensor([torch.Tensor(img_bbox).float().unsqueeze(0) for bbx_uint32 in img_bbox_uint32]) #img_bbox_uint32]) #torch.Tensor(img_bbox_uint32).float().unsqueeze(0)
            #img_label = img_df["category_id"].values.tolist()
            #target["labels"] = torch.Tensor([self.label2target[i] for i in img_label]).long()#.to(device)
            target["labels"] = torch.Tensor(img_bbox).long().unsqueeze(0)#([target for target in self.target2label.keys()]).long()
        
        preprocess_img = self.preprocess_fun(img)
        if self.return_image_path:
            return preprocess_img, target, img_file_path
        else:
            return preprocess_img, target
    
    def get_height_and_width(self):
        return self.orig_height, self.orig_width
    
    def collate_fn(self, batch):
        return tuple(zip(*batch))
    
    def __len__(self):
        return len(self.img_info)
# %%
