
#%%
import pandas as pd


#%%
train_df = pd.read_csv("/home/lin/codebase/hse_count_prediction/zindi_hse_no_pred/Train.csv")

# %%
train_df.category_id.value_counts()
# %%
added visz for tif