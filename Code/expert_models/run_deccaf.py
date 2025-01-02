# %%
import pandas as pd
import numpy as np
import os
import yaml
import hpo
import pickle


data_cfg_path = '../alert_data/dataset_cfg.yaml'

with open(data_cfg_path, 'r') as infile:
    data_cfg = yaml.safe_load(infile)


CATEGORICAL_COLS = data_cfg['data_cols']['categorical']

cat_dict = data_cfg['categorical_dict']

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype != 'category':
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
        elif new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data



expert_ids_path = f'../../FiFAR/synthetic_experts/expert_ids.yaml'

with open(expert_ids_path, 'r') as infile:
    EXPERT_IDS = yaml.safe_load(infile)

cat_dict['assignment'] = EXPERT_IDS['human_ids']
l = 0.057

for train in os.listdir(f'../../FiFAR/testbed/train_alert'):
    train_set = pd.read_parquet(f'../../FiFAR/testbed/train_alert/{train}/train.parquet')
    train_set = train_set.loc[train_set["assignment"] != 'classifier_h']
    train_set = cat_checker(train_set, data_cfg['data_cols']['categorical'] + ['assignment'], cat_dict)

    print(f'Fitting start for {train}, conjoined')
    val_set = train_set.loc[train_set['month'] == 6]
    train_set = train_set.loc[train_set['month'] != 6]
    train_w = train_set['fraud_bool'].replace([0,1],[l,1])
    val_w = val_set['fraud_bool'].replace([0,1],[l,1])
    train_x = train_set.drop(columns = ['fraud_bool', 'batch','month', 'decision'])
    val_x = val_set.drop(columns = ['fraud_bool', 'batch','month', 'decision'])

    train_y = (train_set['decision'] == train_set['fraud_bool']).astype(int)
    val_y = (val_set['decision'] == val_set['fraud_bool']).astype(int)

    if not (os.path.exists(f'../../l2d_benchmarking/expert_models/deccaf/{train}/')):
        os.makedirs(f'../../l2d_benchmarking/expert_models/deccaf/{train}/')

    if not (os.path.exists(f'../../l2d_benchmarking/expert_models/deccaf/{train}/best_model.pickle')):
        opt = hpo.HPO(train_x,val_x,train_y,val_y,train_w, val_w, method = 'TPE', path = f'../../l2d_benchmarking/expert_models/deccaf/{train}/')
        opt.initialize_optimizer(CATEGORICAL_COLS, 10)

                
        
test = pd.read_parquet('../../FiFAR/alert_data/processed_data/alerts.parquet')
test = test.loc[test['month'] == 7]
X_test = test.drop(columns = ['fraud_bool','model_score','month']) 

with open(f"../../l2d_benchmarking/classifier_h/selected_model/best_model.pickle", 'rb') as fp:
    classifier_h = pickle.load(fp)
with open(f"../../l2d_benchmarking/classifier_h/selected_model/model_properties.yaml", 'r') as fp:
    classifier_h_properties = yaml.safe_load(fp)

def sig(x):
    return 1/(1+np.exp(-x))

def output(data, model, init_score):
    return sig(model.predict(data,raw_score=True) + init_score)

h_preds = output(X_test, classifier_h, classifier_h_properties['init_score'])

X_test = test.drop(columns = ['month','fraud_bool'])

preds = dict()

for env in os.listdir(f'../../l2d_benchmarking/expert_models/deccaf/'):
    train_set = pd.read_parquet(f'../../FiFAR/testbed/train_alert/{env}/train.parquet')
    table = pd.DataFrame(index = test.index)
        
    with open(f"../../l2d_benchmarking/expert_models/deccaf/{env}/best_model.pickle", "rb") as input_file:
        model = pickle.load(input_file)
    
    for expert in train_set['assignment'].unique():
        print(expert)
        X_test['assignment'] = expert

        X_test = cat_checker(X_test, data_cfg['data_cols']['categorical'] + ['assignment'], cat_dict)

        table.loc[:,expert] = model.predict_proba(X_test)[:,1]
    
   
    table.loc[:,'classifier_h'] = np.maximum(h_preds,  1-h_preds)
    preds[env] = table

os.makedirs('../../l2d_benchmarking/deferral/l2d_predictions', exist_ok=True)

with open(f"../../l2d_benchmarking/deferral/l2d_predictions/deccaf.pkl", "wb") as out_file:
    pickle.dump(preds, out_file)




