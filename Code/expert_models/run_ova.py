# %%
import pandas as pd
import numpy as np
import os
import yaml
import hpo
import pickle

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data

def sig(x):
    return 1/(1+np.exp(-x))

def output(data, model, init_score):
    return sig(model.predict(data,raw_score=True) + init_score)


with open('../alert_data/dataset_cfg.yaml', 'r') as infile:
    data_cfg = yaml.safe_load(infile)


for train in os.listdir(f'../../FiFAR/testbed/train_alert'):
    train_set = pd.read_parquet(f'../../FiFAR/testbed/train_alert/{train}/train.parquet')
    train_set = train_set.loc[train_set["assignment"] != 'classifier_h']
    experts = train_set['assignment'].unique()
    for expert in experts:
        train_exp = train_set.loc[train_set['assignment'] == expert]
        train_exp = cat_checker(train_exp, data_cfg['data_cols']['categorical'], data_cfg['categorical_dict'])

        val_exp = train_exp.loc[train_exp['month'] == 6]
        train_exp = train_exp.loc[train_exp['month'] != 6]

        train_w = train_exp['fraud_bool'].replace([0,1],[data_cfg['lambda'],1])
        val_w = val_exp['fraud_bool'].replace([0,1],[data_cfg['lambda'],1])
        train_x = train_exp.drop(columns = ['fraud_bool', 'batch','month', 'assignment', 'decision'])
        val_x = val_exp.drop(columns = ['fraud_bool', 'batch','month', 'assignment', 'decision'])

        train_y = (train_exp['decision'] == train_exp['fraud_bool']).astype(int)
        val_y = (val_exp['decision'] == val_exp['fraud_bool']).astype(int)

        if not (os.path.exists(f'../../l2d_benchmarking/expert_models/ova/{train}/{expert}/')):
            os.makedirs(f'../../l2d_benchmarking/expert_models/ova/{train}/{expert}/')

        if not (os.path.exists(f'../../l2d_benchmarking/expert_models/ova/{train}/{expert}/best_model.pickle')):
            opt = hpo.HPO(train_x,val_x,train_y,val_y,train_w, val_w, method = 'TPE', path = f'../../l2d_benchmarking/expert_models/ova/{train}/{expert}/')
            opt.initialize_optimizer(data_cfg['data_cols']['categorical'], 10)


test = pd.read_parquet('../../FiFAR/alert_data/processed_data/alerts.parquet')
test = test.loc[test['month'] == 7]
X_test = test.drop(columns = ['fraud_bool','model_score','month']) 

with open(f"../../l2d_benchmarking/classifier_h/selected_model/best_model.pickle", 'rb') as fp:
    classifier_h = pickle.load(fp)
with open(f"../../l2d_benchmarking/classifier_h/selected_model/model_properties.yaml", 'r') as fp:
    classifier_h_properties = yaml.safe_load(fp)

h_preds = output(X_test, classifier_h, classifier_h_properties['init_score'])

test = cat_checker(test, data_cfg['data_cols']['categorical'], data_cfg['categorical_dict'])
test = test.loc[test['month'] == 7]
X_test = test.drop(columns = ['month','fraud_bool'])

preds = dict()

for env in os.listdir(f'../../l2d_benchmarking/expert_models/ova/'):
    table = pd.DataFrame(index = test.index, columns = os.listdir(f'../../l2d_benchmarking/expert_models/ova/{env}'))
    for expert in os.listdir(f'../../l2d_benchmarking/expert_models/ova/{env}'):
        
        with open(f"../../l2d_benchmarking/expert_models/ova/{env}/{expert}/best_model.pickle", "rb") as input_file:
            model = pickle.load(input_file)
        
        table.loc[:, expert] = model.predict_proba(X_test)[:,1]
    
    table.loc[:,'classifier_h'] = np.maximum(h_preds,  1-h_preds)
    preds[env] = table

os.makedirs('../../l2d_benchmarking/deferral/l2d_predictions', exist_ok=True)

with open(f"../../l2d_benchmarking/deferral/l2d_predictions/ova.pkl", "wb") as out_file:
    pickle.dump(preds, out_file)


