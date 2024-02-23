# %%
import pandas as pd
import yaml
import numpy as np
import hpo_fpr
import os
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data

def fpr_thresh(y_true, y_pred, fpr):
    results = pd.DataFrame()
    results["true"] = y_true
    results["score"] = y_pred
    temp = results.sort_values(by="score", ascending=False)

    N = (temp["true"] == 0).sum()
    FP = round(fpr * N)
    aux = temp[temp["true"] == 0]
    threshold = aux.iloc[FP - 1, 1]
    y_pred = np.where(results["score"] >= threshold, 1, 0)
    tpr = metrics.recall_score(y_true, y_pred)

    return tpr, threshold


with open((Path(__file__).parent/'../alert_data/dataset_cfg.yaml').resolve(), 'r') as infile:
    data_cfg = yaml.safe_load(infile)

# DATA LOADING -------------------------------------------------------------------------------------
data = pd.read_csv(Path(__file__).parent/'../../FiFAR/alert_data/Base.csv')
LABEL_COL = data_cfg['data_cols']['label']
TIMESTAMP_COL = data_cfg['data_cols']['timestamp']
PROTECTED_COL = data_cfg['data_cols']['protected']
CATEGORICAL_COLS = data_cfg['data_cols']['categorical']

data.sort_values(by = 'month', inplace = True)
data.reset_index(inplace=True)
data.drop(columns = 'index', inplace = True)
data.index.rename('case_id', inplace=True)
data.loc[:,data_cfg['data_cols']['categorical']] = data.loc[:,data_cfg['data_cols']['categorical']].astype('category')
data = cat_checker(data, data_cfg['data_cols']['categorical'], data_cfg['categorical_dict'])

train = data.loc[(data["month"] < 3)].drop(columns="month")
ml_val = data.loc[(data["month"] == 3)].drop(columns="month")
deployment = data.loc[(data["month"] > 2)].drop(columns="month")

X_train = train.drop(columns = 'fraud_bool')
y_train = train['fraud_bool']
X_val = ml_val.drop(columns = 'fraud_bool') 
y_val = ml_val['fraud_bool']

if not (os.path.exists(Path(__file__).parent/'../../FiFAR/alert_model/best_model.pickle')):
    opt = hpo_fpr.HPO(X_train,X_val,y_train,y_val, method = 'TPE', path = f"./model")
    opt.initialize_optimizer(CATEGORICAL_COLS, 25)

with open(Path(__file__).parent/'../../FiFAR/alert_model/best_model.pickle', 'rb') as infile:
        model = pickle.load(infile)

y_pred = model.predict_proba(X_val)
y_pred = y_pred[:,1]
roc_curve_clf = dict()
rec_at_5, thresh = fpr_thresh(y_val, y_pred, 0.05)

os.makedirs('../../FiFAR/alert_data/processed_data/', exist_ok = True)

deployment['model_score'] = model.predict_proba(deployment.drop(columns = 'fraud_bool'))[:,1]
deployment.to_parquet(Path(__file__).parent/'../../FiFAR/alert_data/processed_data/BAF_alert_model_score.parquet')

model_properties = {'fpr':0.05,
                    'fnr': 1 - rec_at_5,
                    'threshold': thresh
                    }


file_to_store = open(Path(__file__).parent/"../../FiFAR/alert_model/model_properties.pickle", "wb")
pickle.dump(model_properties, file_to_store)
file_to_store.close()


