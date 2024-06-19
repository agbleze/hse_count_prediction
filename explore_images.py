
#%%
import pandas as pd
import numpy as np
import rasterio
import os
import ast

#%%
train_path = "/home/lin/codebase/hse_count_prediction/zindi_hse_no_pred/Train.csv"
train_df = pd.read_csv(train_path)

# %%
train_df.category_id.value_counts()

#%%
train_df.info()

#%%
train_df.describe()

#%%

train_df.head()

#%%

train_df.image_id.nunique()

#%%
train_df.id.nunique()
#%%
(train_df.groupby(by=["image_id", "category_id"])[["category_id"]]
 .count().rename(columns={"category_id": "total_obj"})
 .reset_index()
 )


#%%
imgs_without_bbox = train_df[train_df["bbox"].isna()]["image_id"].unique().tolist()

#%%

train_df.isna()#.sum()
train_df[train_df["bbox"].isna()]
#%%
from PIL import Image
import cv2

#%%
img_path = "/home/lin/codebase/hse_count_prediction/zindi_hse_no_pred/Images/id_0b0pzumg4rbl.tif"
img = Image.open(img_path)

#%%
img.size

#%%
import ast

ast.literal_eval(train_df.loc[40].category_id)

#%%
train_df.loc["image_id"]
#%%
type(train_df.loc[40].bbox)


#%%
train_df[train_df.image_id=="id_zzg2s87ku9nn"].bbox.to_list()

#%%

img_cvread = cv2.imread(img_path)

#%%
cv2.cvtColor(img_cvread, cv2.COLOR_BGR2RGB)

category_map = {1: "Other", 2: "Tin", 3: "Thatch"}

#%%
def visualize_bboxes(annotation_file, image_dir, output_dir,
                     category_map=category_map,
                     img_ext="tif"
                    ):
    os.makedirs(output_dir, exist_ok=True)
    # Load COCO annotation file
    annot_df = pd.read_csv(annotation_file)
    
    for img_id in annot_df.image_id.unique():
        
        img_id_df = annot_df[annot_df.image_id == img_id]
        bbox_lst_str = img_id_df.bbox.to_list()
        category_id_list = img_id_df.category_id.to_list()
        bbox = []
        for bbx in bbox_lst_str:
            if isinstance(bbx, str):
                bbox.append(ast.literal_eval(bbx))
                #category_id = df_ind.category_id
            else:
                bbox = None
        img_name_with_ext = f"{img_id}.{img_ext}"
        image_path = os.path.join(image_dir, img_name_with_ext)
        output_path = os.path.join(output_dir, f"bbox_{img_name_with_ext}")
        
        image = cv2.imread(image_path)
        
        if not bbox:
            cv2.imwrite(output_path, image)
            print(f"Bounding boxes visualized for {img_name_with_ext} and saved as {output_path}")
        else:
            print(bbox)
            print(category_id_list)
            for bbx, category_id in zip(bbox, category_id_list):
                print(bbx)
                x, y, w, h = map(int, bbx)
                category_name = category_map.get(category_id)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, category_name, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                            )
            cv2.imwrite(output_path, image)
            print(f"Bounding boxes visualized for {img_name_with_ext} and saved as {output_path}")

#%%
img_dir = "/home/lin/codebase/hse_count_prediction/zindi_hse_no_pred/Images"
visualize_bboxes(annotation_file=train_path, image_dir=img_dir,
                 category_map=category_map,
                 output_dir="visualize_bbox"
                 )

# %%
#1 added visz for tif
#2 

def get_tiff_img(path, return_all_bands, bands=("B01", "B03", "B02"),
                 normalize_bands=True
                ):
    all_band_names = ("B01","B02", "B03","B04","B05", "B06",
                      "B07","B08","B8A","B09","B11","B12"
                    )
    if return_all_bands:
        band_indexs = [all_band_names.index(band_name) for band_name in all_band_names]
    
    else:
        band_indexs = [all_band_names.index(band_name) for band_name in bands]
    #print(band_indexs)
    with rasterio.open(path) as src:
        img_bands = [src.read(band) for band in range(1,13)]
    dstacked_bands = np.dstack([img_bands[band_index] for band_index in band_indexs])
    #dstacked_bands = np.dstack([img_bands[3], img_bands[2], img_bands[1]])
    if normalize_bands:
        # Normalize bands to 0-255
        dstacked_bands = ((dstacked_bands - dstacked_bands.min()) / 
                          (dstacked_bands.max() - dstacked_bands.min()) * 255
                          ).astype(np.uint8)

    return dstacked_bands