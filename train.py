import torch
from torch_snippets import find, show
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from copy import deepcopy
import os
import numpy as np
from glob import glob
import cv2
from dataset_obj import preprocess_img
import pandas as pd
from torchvision.models.detection.faster_rcnn import (FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                                                      FasterRCNN, 
                                                      FasterRCNN_ResNet50_FPN_Weights
                                                      )
from torchvision.models.resnet import ResNet50_Weights, resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from typing import Any, Optional
from torchvision.models._utils import _ovewrite_value_param
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection._utils import retrieve_out_channels
from torchvision.models.detection.ssd import SSDHead

device = "cuda" if torch.cuda.is_available() else "cpu"

#device = "cpu"
backbone_kwargs = [
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    "ssd300_vgg16"
]
def get_model(num_classes, backbone,
              trainable_backbone_layers=3,
              **kwargs,
              ):
    assert (backbone in backbone_kwargs), f"{backbone} is not in one of the backone that can be called. Should be one of {backbone_kwargs}"
    
    if backbone == "fasterrcnn_resnet50_fpn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                     trainable_backbone_layers=trainable_backbone_layers
                                                                     )
    elif backbone == "fasterrcnn_resnet50_fpn_v2":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True,
                                                                        trainable_backbone_layers=trainable_backbone_layers
                                                                        )
    elif backbone == "fasterrcnn_mobilenet_v3_large_fpn":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True,
                                                                               trainable_backbone_layers=trainable_backbone_layers
                                                                               )
    elif backbone == "fasterrcnn_mobilenet_v3_large_320_fpn":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True,
                                                                                   trainable_backbone_layers=trainable_backbone_layers
                                                                                   ) 
    elif backbone == "ssd300_vgg16":
        model = torchvision.models.detection.ssd300_vgg16(pretrained=True,
                                                          trainable_backbone_layers=trainable_backbone_layers,
                                                          **kwargs
                                                          ) 
        #ssdmodel = torchvision.models.detection.ssd300_vgg16(pretrained=True)
        num_anchors = model.anchor_generator.num_anchors_per_location()
        ssdbackbone = model.backbone
        out_channels = retrieve_out_channels(model=ssdbackbone, size=(300,300))
        head = SSDHead(in_channels=out_channels, num_anchors=num_anchors,
                        num_classes=num_classes
                        )
        model.head = head
        
    if backbone != "ssd300_vgg16":  
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, 
                                                        num_classes=num_classes
                                                        )
    return model.to(device)

def train_batch(inputs, model, optimizer):
    model.to(device).train()
    input, targets = inputs
    #print(f"targets: {targets}")
    input = list(image.to(device) for image in input)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    #print(f"targets. {targets}")
    optimizer.zero_grad()
    losses = model(input, targets)
    loss = sum(loss for loss in losses.values())
    loss.backward()
    optimizer.step()
    return loss, losses, model, optimizer

@torch.no_grad()
def validate_batch(inputs, model, optimizer):
    model.train()
    input, targets = inputs
    inputs = list(image.to(device) for image in input)
    targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
    optimizer.zero_grad()
    losses = model(inputs, targets)
    loss = sum(loss for loss in losses.values())
    return loss, losses

def train_ssd(trn_dataloader, test_dataloader, 
              model, optimizer,
            n_epochs, log, model_store_dir,
            model_name
            ):
    for epoch in range(n_epochs):
        _n = len(trn_dataloader)
        for ix, inputs in enumerate(trn_dataloader):
            model.to(device).train()
            input, targets = inputs
            inputs = list(image.to(device) for image in input)
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
            optimizer.zero_grad()
            losses = model(inputs, targets)
            bbox_regression, classification = losses["bbox_regression"], losses["classification"]
            pos = (epoch + (ix+1)/_n)
            trn_loss = bbox_regression + classification
            log.record(pos, trn_loss=trn_loss, 
                       trn_regr_loss=bbox_regression,
                       trn_loc_loss=classification,
                       end="\r"
                       )
        _n = len(test_dataloader)
        for ix, inputs in enumerate(test_dataloader):
            val_loss, val_losses = validate_batch(inputs, model, optimizer)
            val_bbox_regression, val_classification = val_losses["bbox_regression"], val_losses["classification"]
            pos = (epoch + (ix+1)/_n)
            log.record(pos, val_loss=val_loss.item(), val_loc_loss=val_classification,
                       val_regr_loss=val_bbox_regression,
                       end="\r"
                       )
            
        if (epoch+1)%(n_epochs/n_epochs)==0:
            log.report_avgs(epoch+1)
            print("saving model as dict")
            model_path = os.path.join(model_store_dir, f'{model_name}_epoch_{epoch+1}.pth')
            torch.save(deepcopy(model.to("cpu").state_dict()), model_path)
            
            # save model in state for infernece / resuming training
            print("saving model as checkpoint")
            resume_model_path = os.path.join(model_store_dir, 
                                             f'{model_name}_resumable_epoch_{epoch+1}.pth'
                                             )
            torch.save({"epoch": epoch+1,
                        "model_state_dict": deepcopy(model.to("cpu").state_dict()),
                        "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                        "val_loss": deepcopy(val_loss),
                        },
                       resume_model_path
                       )
            
            # save model as torchscript file for easy loading
            print("Exporting to torchscript")
            torchscript_model_path = os.path.join(model_store_dir, 
                                             f'{model_name}_torchscript_epoch_{epoch+1}.pt'
                                             )
            model_scripted = torch.jit.script(deepcopy(model.to("cpu")))
            model_scripted.save(torchscript_model_path)
    return log
    
    
#%% 
def trigger_train(trn_dataloader, test_dataloader, model, optimizer,
                  n_epochs, log, model_store_dir,
                  model_name
                  ):
    trained_models_list = [model]
    optimizer_list = [optimizer]
    for epoch in range(n_epochs):
        _n = len(trn_dataloader)
        for ix, inputs in enumerate(trn_dataloader):
            if epoch == 0:
                loss, losses, model_trained, optimizer_trained = train_batch(inputs, trained_models_list[0], 
                                                                            optimizer_list[0]
                                                                            )
                trained_models_list.append(model_trained)
                optimizer_list.append(optimizer_trained)
            else:
                loss, losses, model_trained, optimizer_trained = train_batch(inputs, trained_models_list[-1], 
                                                          optimizer_list[-1]
                                                          )
                print(f"loss: {loss}")
                print(f"losses: {losses}")
                trained_models_list.append(model_trained)
                optimizer_list.append(optimizer_trained)
                trained_models_list.pop(0)
                optimizer_list.pop(0)
            loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = (
                [losses[k] for k in ["loss_classifier", "loss_box_reg",
                                    "loss_objectness", "loss_rpn_box_reg"
                                    ]
                ]
            )
            pos = (epoch + (ix+1)/_n)
            log.record(pos, trn_loss=loss.item(), trn_loc_loss=loc_loss.item(),
                       trn_regr_loss=regr_loss.item(),
                       trn_objectness_loss=loss_objectness.item(),
                       trn_rpn_box_reg_loss=loss_rpn_box_reg.item(),
                       end="\r"
                       )
            
        _n = len(test_dataloader)
        for ix, inputs in enumerate(test_dataloader):
            loss, losses = validate_batch(inputs, model_trained, optimizer_trained)
            loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = (
                [losses[k] for k in ["loss_classifier", "loss_box_reg", "loss_objectness",
                                     "loss_rpn_box_reg"
                                     ]
                 ]
            )
            pos = (epoch + (ix+1)/_n)
            log.record(pos, val_loss=loss.item(), val_loc_loss=loc_loss.item(),
                       val_regr_loss=regr_loss.item(), val_objectness=loss_objectness.item(),
                       val_rpn_box_reg_loss=loss_rpn_box_reg.item(),
                       end="\r"
                       )
        if (epoch+1)%(n_epochs/n_epochs)==0:
            log.report_avgs(epoch+1)
            print("saving model as dict")
            model_path = os.path.join(model_store_dir, f'{model_name}_epoch_{epoch+1}.pth')
            torch.save(deepcopy(model_trained.to("cpu").state_dict()), model_path)
            
            # save model in state for infernece / resuming training
            print("saving model as checkpoint")
            resume_model_path = os.path.join(model_store_dir, 
                                             f'{model_name}_resumable_epoch_{epoch+1}.pth'
                                             )
            torch.save({"epoch": epoch+1,
                        "model_state_dict": deepcopy(model_trained.to("cpu").state_dict()),
                        "optimizer_state_dict": deepcopy(optimizer_trained.to("cpu").state_dict()),
                        "loss": deepcopy(loss),
                        },
                       resume_model_path
                       )
            
            # save model as torchscript file for easy loading
            print("Exporting to torchscript")
            torchscript_model_path = os.path.join(model_store_dir, 
                                             f'{model_name}_torchscript_epoch_{epoch+1}.pt'
                                             )
            model_scripted = torch.jit.script(deepcopy(model_trained.to("cpu")))
            model_scripted.save(torchscript_model_path)
            
    return log


def decode_output(output, nms_threshold):
    bbs = output["boxes"].cpu().detach().numpy().astype(np.uint16)
    labels = np.array([target for target in output["labels"].cpu().detach().numpy()])
    confs = output["scores"].cpu().detach().numpy()
    ixs = nms(torch.tensor(bbs.astype(np.float32)), 
              torch.tensor(confs), 
              nms_threshold
              )
    bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]
    
    if len(ixs) == 1:
        bbs, confs, labels = [np.array([tensor]) for tensor in [bbs, confs, labels]]
    return bbs.tolist(), confs.tolist(), labels.tolist()



def predict(model, test_dataloader,
            device, num_images,
            nms_threshold,
            save_dir=None, 
            multiple_pred_save_dir=None
            ):
    model.eval().to(device)
    for ix, (images, targets, img_paths) in enumerate(test_dataloader):
        
        if num_images:
            #print(f"num_images: {num_images}")
            if ix==num_images: break
        images = [im.to(device) for im in images]
        outputs = model(images)
        for ix, output in enumerate(outputs):
            bbs, confs, labels = decode_output(output, 
                                               nms_threshold=nms_threshold
                                               )
            max_conf_index = confs.index(max(confs))
            bbs, confs, labels = [bbs[max_conf_index]], confs[max_conf_index], labels[max_conf_index]
            print(f"bbs: {bbs}  confs: {confs}")
            print(f"len(bbs): {len(bbs)}")
            #info = [f"{l}@{c:2f}" for l, c in zip(labels, confs)]
            info = [f"{labels}@{confs:2f}"]
            
            if len(bbs) > 1 or len(bbs) == 0:
                save_dir = multiple_pred_save_dir
            elif len(bbs) == 1:
                save_dir = save_dir
                
            if save_dir:
                img_path = os.path.basename(img_paths[ix])
                save_path = os.path.join(save_dir, img_path)
                print(save_path)
                show(images[ix].cpu().permute(1,2,0), bbs=bbs, texts=info, 
                    sz=25, #title=info, 
                    save_path=save_path
                    )
            else:
                show(images[ix].cpu().permute(1,2,0), 
                     bbs=bbs, texts=info, 
                    sz=15, #title=info
                    )


def predict_img(img_path_list, model, device, 
                preprocess_fun=preprocess_img,
                image_name=None, export_dir=None,
                img_dir=None,
                return_result_as_df=True,
                ):
    """Takes an image and makes a prediction or a directory
    and make prediction on all images there
    """
    model.eval().to(device)
    if not img_path_list:
        if not img_dir:
            raise ValueError(f"img_dir cannot be {img_dir} when img_path_list is not given")
        img_path_list = glob(f"{img_dir}/*")
    for img_path in img_path_list:
        image_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)/255
        preprocess_img = preprocess_fun(img)
        outputs = model(preprocess_img)
        export_img_path = os.path.join(export_dir, image_name)
        for ix, output in enumerate(outputs):
            bbs, confs, labels = decode_output(output)
            info = [f"{l}@{c:2f}" for l, c in zip(labels, confs)]
            show(preprocess_img[ix].cpu().permute(1,2,0), 
                 bbs=bbs, 
                 texts=info, 
                 sz=5, title=image_name,
                 save_path=export_img_path
                 )
        

def predict_bbox(model, test_dataloader,
                 submission_df,
                device, 
                nms_threshold
                ) -> pd.DataFrame:
    model.eval().to(device)
    for ix, (images, img_paths) in enumerate(test_dataloader):
        image_id = os.path.basename(img_paths).split(".")[0]
        outputs = model(images)
        for ix, output in enumerate(outputs):
            bbs, confs, labels = decode_output(output, 
                                               nms_threshold=nms_threshold
                                               )
            max_conf_index = confs.index(max(confs))
            bbs, confs, labels = [bbs[max_conf_index]], confs[max_conf_index], labels[max_conf_index]
            bbox = [int(x) for x in bbs]
            submission_df.loc[image_id] = bbox
    return submission_df
                                

#%%
def fasterrcnn_resnet50_fpn_load_from_local(weights: Optional[FasterRCNN_ResNet50_FPN_Weights] = None,
                                            progress: bool = True,
                                            num_classes: Optional[int] = None,
                                            weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
                                            trainable_backbone_layers: Optional[int] = None,
                                            weight_path: str = None
                                            ):
    weights = FasterRCNN_ResNet50_FPN_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

    backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes=num_classes)
    model.load_state_dict(torch.load(weight_path))
    return model

    # if weights is not None:
    #     model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    #     if weights == FasterRCNN_ResNet50_FPN_Weights.COCO_V1:
    #         overwrite_eps(model, 0.0)
#%%
if __name__ == "__main__":
    
    #%%
    import pandas as pd
    from dataset_obj import SpaceCraftDataset
    from sklearn.model_selection import train_test_split

    #%%
    all_img_dir = "/home/lin/codebase/spacecraft_detection/spacecraft_detection/all_images/"
    weight_path = "/home/lin/codebase/spacecraft_detection/model_store/spacecraft_onlybbox_epoch_9.pth"


    train_label_path = "/home/lin/codebase/spacecraft_detection/spacecraft_detection/data/train_labels.csv"

    train_metadata_path = "/home/lin/codebase/spacecraft_detection/spacecraft_detection/data/train_metadata.csv"
    train_df = pd.read_csv(train_label_path)
    train_metadata_df = pd.read_csv(train_metadata_path)


    all_df = train_df.merge(train_metadata_df,
                            left_on="image_id",
                            right_on="image_id"
                            )

    all_df["spacecraft_bg_id"] = all_df["spacecraft_id"].astype(str) + '_' + all_df["background_id"].astype(str)

    all_df["only_spacecraft"] = "spacecraft"
    trn, test = train_test_split(all_df, test_size=0.3, random_state=2024,
                                stratify=all_df["spacecraft_id"]
                                )


    val, test  = train_test_split(test, test_size=0.1, random_state=2024,
                                stratify=test["spacecraft_id"]
                                )

    #%%
    test_prediction_dir = "/home/lin/codebase/spacecraft_detection/test_predictions"
    test_data = SpaceCraftDataset(df=test, img_dir=all_img_dir,
                                only_bbox=True, resize=None,
                                return_image_path=True
                                )

    test_dataloader = DataLoader(dataset=test_data, batch_size=1, 
                                drop_last=False,
                                collate_fn=test_data.collate_fn,
                                
                                )

    #%%
    wrong_pred_dir = "/home/lin/codebase/spacecraft_detection/wrong_detections"
    nms_pred0001_dir = "/home/lin/codebase/spacecraft_detection/nms_pred0001"
    nms_pred0001_max_conf = "/home/lin/codebase/spacecraft_detection/nms_pred0001_max_conf"
    max_conf_pred = "/home/lin/codebase/spacecraft_detection/max_conf_pred"

    save_dir = "/home/lin/codebase/spacecraft_detection/reloadmodel_pred/reload_save_dir"
    multipred_dir = "/home/lin/codebase/spacecraft_detection/reloadmodel_pred/multipred_dir"
    #%%
    preloaded_model = fasterrcnn_resnet50_fpn_load_from_local(weight_path=weight_path,
                                            num_classes=2)



    #%%

    test_data_split = pd.read_csv("/home/lin/codebase/spacecraft_detection/test_data_split.csv",
                                index_col="image_id"
                                )[["xmin", "ymin", "xmax", "ymax"]]

    #%%
    test_data_split.to_csv("testingsplit_gt.csv")

    #%%

    for ind in test_data_split.index:
        test_data_split.loc[ind] = [0,0,0,0]

    #%%
    testing_img_paths = [img_path for img_path in glob(f"{all_img_dir}/*") 
                        if os.path.basename(img_path).split(".")[0] 
                        in test_data_split.index
                        ]
    #%%
    from example_benchmark.main import predict_bbox, SpacecraftPredictDataset
    
    #%%
    testing_data = SpacecraftPredictDataset(img_dir=all_img_dir,
                                            files=testing_img_paths
                                            )

    testing_dataloader = DataLoader(dataset=testing_data, batch_size=1, 
                                drop_last=False,
                                collate_fn=test_data.collate_fn,
                                
                                )


    #%%
    testing_pred_df = predict_bbox(model=preloaded_model, 
                                    test_dataloader=testing_dataloader,
                                    submission_df=test_data_split, 
                                    device="cpu", 
                                    nms_threshold=0.0001
                                    )

    #%%

    testing_pred_df.to_csv("testingsplit_pred.csv")
    #%%

    predict(model=preloaded_model, test_dataloader=test_dataloader, 
            num_images=None,
            device="cuda",
            save_dir=save_dir,
            multiple_pred_save_dir=multipred_dir,
            nms_threshold=0.0001
            )

    #%%
    import copy
    model_scripted = torch.jit.script(copy.deepcopy(preloaded_model))
    model_scripted.save("model_scripted.pt")
    
    #%%
    script_model_reloaded = torch.jit.load("/home/lin/codebase/spacecraft_detection/model_store/spacecraft_224X224_fasterrcnn_mobilenet_v3_large_fpn_torchscript_epoch_10.pt",
                                           map_location="cpu")
    
    #%%
    print(torch.jit.script(script_model_reloaded).code)
    
    
    #%%  #####   testing of full data   #######

    full_data_labels = pd.read_csv(train_label_path, index_col="image_id")[["xmin","ymin","xmax","ymax"]]


    full_data_labels.to_csv("full_data_labels.csv")

    #%%
    for ind in full_data_labels.index:
        full_data_labels.loc[ind] = [0,0,0,0]

    #%%
    fulldata_img_paths = [img_path for img_path in glob(f"{all_img_dir}/*") 
                        if os.path.basename(img_path).split(".")[0] 
                        in full_data_labels.index
                        ]
    #%%
    from example_benchmark.main import predict_bbox, SpacecraftPredictDataset
    full_dataset = SpacecraftPredictDataset(img_dir=all_img_dir,
                                            files=fulldata_img_paths,
                                            device="cpu"
                                            )

    fulldata_dataloader = DataLoader(dataset=full_dataset, batch_size=1, 
                                    drop_last=False,
                                    collate_fn=full_dataset.collate_fn,
                                    num_workers=2,
                                    )

    #%%
    from example_benchmark.main import predict_bbox
    fulldata_pred_df = predict_bbox(model=script_model_reloaded, #preloaded_model,
                                    test_dataloader=fulldata_dataloader,
                                    submission_df=full_data_labels, 
                                    device="cpu", 
                                    nms_threshold=0.7,
                                    using_torchscript_weight=True
                                    )
    # UserWarning: RCNN always returns a (Losses, Detections) tuple in scriptin
    #%%
    fulldata_pred_df.to_csv("fulldata_pred.csv")
    
    #%% ########## model quant   ##
    from example_benchmark.main import load_model
    model_path = "example_benchmark/spacecraft_onlybbox_epoch_9.pth"
    #model_path = "example_benchmark/spacecraft_224X224_epoch_4.pth"
    resnetweight_path = "example_benchmark/resnet50-0676ba61.pth"
    model = load_model(weight_path=model_path,
                        num_classes=2, 
                        resnetweight_path=resnetweight_path,
                        device=device
                        )
    
    #%%