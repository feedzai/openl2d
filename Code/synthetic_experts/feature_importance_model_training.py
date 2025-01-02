import yaml
import pandas as pd
import hpo
import os
import pickle


data_cfg_path = '../alert_data/dataset_cfg.yaml'


with open(data_cfg_path, 'r') as infile:
    data_cfg = yaml.safe_load(infile)


cat_dict = data_cfg['categorical_dict']

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data

data = pd.read_parquet(f'../../FiFAR/alert_data/processed_data/alerts.parquet')
preds = pd.read_parquet(f'../../FiFAR/synthetic_experts/expert_predictions.parquet').loc[data.index]

LABEL_COL = data_cfg['data_cols']['label']
TIMESTAMP_COL = data_cfg['data_cols']['timestamp']
PROTECTED_COL = data_cfg['data_cols']['protected']
CATEGORICAL_COLS = data_cfg['data_cols']['categorical']

data = cat_checker(data, CATEGORICAL_COLS, cat_dict)

train = data.loc[(data["month"] > 2) & (data["month"] < 6)]
val = data.loc[data["month"] == 6]
test = data.loc[data["month"] == 7]

X_train = train.drop(columns = ['fraud_bool','month']).sample(6000, random_state = 42)
X_val = val.drop(columns = ['fraud_bool','month']).sample(2000, random_state = 42)
X_test = test.drop(columns = ['fraud_bool','month']).sample(2000, random_state = 42)

roc_curves_val = dict()
roc_curves_test = dict()
roc_auc = dict()
best_thresh = dict()


for expert in preds.columns:
    y_train = preds.loc[X_train.index,expert]
    y_val = preds.loc[X_val.index,expert]
    opt = hpo.HPO(X_train,X_val,y_train,y_val, method = 'TPE', path = f"../../l2d_benchmarking/synthetic_experts/feature_dependence_models/{expert}")
    if not (os.path.exists(f'../../l2d_benchmarking/feature_dependence_models/{expert}/best_model.pickle')):
        opt.initialize_optimizer(CATEGORICAL_COLS, 10)