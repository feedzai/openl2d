# %%
import pandas as pd
import yaml
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer, StandardScaler
from sklearn.metrics import confusion_matrix
import pickle
import expert_src as experts
import numpy as np
import os
from pathlib import Path
import shutil

def sample(mu,sigma,n,prev_at_5, l, top_clip, bottom_clip, max_FPR, min_FPR, max_FNR, min_FNR):

    slope = -(prev_at_5)/(l*(1-prev_at_5))
    costs = np.random.normal(loc = mu, scale = sigma, size = n)
    costs = np.clip(costs, bottom_clip, top_clip)
    experts = []

    for cost in costs:
        line = pd.DataFrame()
        line['x'] = np.random.uniform(0.0001,0.9999,size = 10000)
        line['y'] = line['x']*slope + cost/(l*(1-prev_at_5))
        line = line.loc[(line['y']>=min_FPR) & (line['y']<=max_FPR)]
        line = line.loc[(line['x']>=min_FNR) & (line['x']<=max_FNR)]
        selec = np.random.choice(line.index)
        experts.append([line['x'].loc[selec],line['y'].loc[selec]])
    
    experts = np.array(experts)
    return experts

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            print(f'{feature} has been reencoded')
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data

data_cfg_path = Path(__file__).parent/'../alert_data/dataset_cfg.yaml'
with open(data_cfg_path, 'r') as infile:
    data_cfg = yaml.safe_load(infile)

cfg_path = Path(__file__).parent/'./cfg.yaml'
with open(cfg_path, 'r') as infile:
    cfg = yaml.safe_load(infile)

cat_dict = data_cfg['categorical_dict']
LABEL_COL = data_cfg['data_cols']['label']
TIMESTAMP_COL = data_cfg['data_cols']['timestamp']
PROTECTED_COL = data_cfg['data_cols']['protected']
CATEGORICAL_COLS = data_cfg['data_cols']['categorical']

#Loading ML Model and its properties
with open(Path(__file__).parent/'../../FiFAR/alert_model/best_model.pickle', 'rb') as infile:
    ml_model = pickle.load(infile)

with open(Path(__file__).parent/'../../FiFAR/alert_model/model_properties.pickle', 'rb') as infile:
    ml_model_properties = pickle.load(infile)

data = pd.read_parquet(f'../../FiFAR/alert_data/processed_data/alerts.parquet')
data = cat_checker(data, CATEGORICAL_COLS, cat_dict)

train_test = data.loc[data["month"] != 6]
val = data.loc[data["month"] == 6]
y_val = val['fraud_bool']

tn, fp, fn, tp = confusion_matrix(y_val, np.ones(len(y_val))).ravel()

l=0.057
cost_rejec_all_val = (fn+fp*l)/(len(y_val))
prev_at_5 = y_val.mean()

# Creating ExpertTeam object. 
expert_team = experts.ExpertTeam()
EXPERT_IDS = dict(human_ids=list())
THRESHOLDS = dict()


ml_model_threshold = ml_model_properties['threshold']
ml_model_recall = 1 - ml_model_properties['fnr']
ml_model_fpr = ml_model_properties['fpr']


#We use the ML Model training split to fit our experts.
#The expert fitting process involves determining the ideal Beta_0 and Beta_1 to obtain the user's desired target FPR and FNR
experts_train_X = val.copy().drop(columns=LABEL_COL)
#Change customer_age variable to a binary
experts_train_X[PROTECTED_COL] = (experts_train_X[PROTECTED_COL] >= 50).astype(int)

#Apply same process to the deployment split
experts_deployment_X = train_test.copy().drop(columns=LABEL_COL)
experts_deployment_X[PROTECTED_COL] = (experts_deployment_X[PROTECTED_COL] >= 50).astype(int)

#Transform the numerical columns into quantiles and subtract 0.5 so they exist in the [-0.5, 0.5] interval
cols_to_quantile = experts_train_X.drop(columns=CATEGORICAL_COLS).columns.tolist()
qt = QuantileTransformer(random_state=42)
experts_train_X[cols_to_quantile] = (
    qt.fit_transform(experts_train_X[cols_to_quantile])
    - 0.5  # centered on 0
)

#Target encode and transform the categorical columns
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
experts_train_X[CATEGORICAL_COLS] = oe.fit_transform(experts_train_X[CATEGORICAL_COLS])

ss = StandardScaler(with_std=False)
experts_train_X[:] = ss.fit_transform(experts_train_X)

cols_to_scale = [c for c in experts_train_X.columns if c not in cols_to_quantile]
desired_range = 1
scaling_factors = (
    desired_range /
    (experts_train_X[cols_to_scale].max() - experts_train_X[cols_to_scale].min())
)
experts_train_X[cols_to_scale] *= scaling_factors

# Preprocess the deployment splits and save the transformed data
def preprocess(df):
    processed_X = df.copy()
    processed_X[cols_to_quantile] = qt.transform(processed_X[cols_to_quantile]) - 0.5  # centered on 0
    processed_X[CATEGORICAL_COLS] = oe.transform(processed_X[CATEGORICAL_COLS])
    processed_X[:] = ss.transform(processed_X)
    processed_X[cols_to_scale] *= scaling_factors
    return processed_X

experts_train_X['month'] = val['month']
experts_deployment_X = preprocess(experts_deployment_X)
experts_deployment_X['month'] = train_test['month']
experts_train_X = experts_train_X.drop(columns = 'month')
experts_deployment_X = experts_deployment_X.drop(columns = 'month')

# Synthetic Expert Generation -----------------------------------------------------------------------------------
#This function allows a user to create other groups by only defining the parameters that differ from the regular experts
def process_groups_cfg(groups_cfg, baseline_name='standard'):
    full_groups_cfg = dict()
    for g_name in groups_cfg:
        if g_name == baseline_name:
            full_groups_cfg[g_name] = groups_cfg[g_name]
        else:
            full_groups_cfg[g_name] = dict()
            for k in groups_cfg[baseline_name]:
                if k not in list(groups_cfg[g_name].keys()):
                    full_groups_cfg[g_name][k] = full_groups_cfg[baseline_name][k]
                elif isinstance(groups_cfg[g_name][k], dict):
                    full_groups_cfg[g_name][k] = {  # update baseline cfg
                        **groups_cfg[baseline_name][k],
                        **groups_cfg[g_name][k]
                    }
                else:
                    full_groups_cfg[g_name][k] = groups_cfg[g_name][k]

    return full_groups_cfg


ensemble_cfg = process_groups_cfg(cfg['experts']['groups'])
expert_properties_list = list()
seeds_set = 0
#For each expert group generate the number of experts
for group_name, group_cfg in ensemble_cfg.items():

    if seeds_set == 0: 
        np.random.seed(group_cfg['group_seed'])
        expert_seeds = np.random.randint(low = 2**32-1, size = group_cfg['n'])
        seeds_set = 1
    
    coefs_gen = dict()

    for coef in ['score', 'protected', 'alpha']:

        if (f'{coef}_mean' in group_cfg.keys()) and (f'{coef}_stdev' in group_cfg.keys()):
            coefs_gen[coef] = np.random.normal(
                    loc=group_cfg[f'{coef}_mean'],
                    scale=group_cfg[f'{coef}_stdev'],
                    size=group_cfg['n']
            )
        else:
            coefs_gen[coef] = np.random.normal(
                    loc=0,
                    scale=0,
                    size=group_cfg['n']
            )
        
    coefs_spe = dict()
    coefs_spe['fnr'] = dict()
    coefs_spe['fpr'] = dict()

    #Generate the set of T_FPR, T_FNR for the group
    generated = sample(group_cfg['cost']['target_mean'],
                       group_cfg['cost']['target_stdev'],
                       group_cfg['n'], 
                       prev_at_5, 
                       l, 
                       top_clip = group_cfg['cost']['top_clip'], 
                       bottom_clip = group_cfg['cost']['bottom_clip'], 
                       min_FNR=group_cfg['cost']['min_FNR'], 
                       max_FNR=group_cfg['cost']['max_FNR'],
                       min_FPR=group_cfg['cost']['min_FPR'], 
                       max_FPR=group_cfg['cost']['max_FPR'])
    
    coefs_spe['fnr']['target'] = generated.T[0]
    coefs_spe['fpr']['target'] = generated.T[1]
    
    #Setting each expert's seed (for sampling of individual feature weights)
    
    if group_cfg['w_dict'] == 'None':
        group_cfg['w_dict'] = None
    
    if 'w_std' not in group_cfg:
        group_cfg['w_std'] = None
    
    if 'theta' not in group_cfg:
        group_cfg['theta'] = None
        
    for i in range(group_cfg['n']):
        expert_name = f'{group_name}#{i}'
        expert_args = dict(
            fnr_target=coefs_spe['fnr']['target'][i],
            fpr_target=coefs_spe['fpr']['target'][i],
            features_w_std = group_cfg['w_std'],
            alpha = coefs_gen['alpha'][i],
            fpr_noise = 0.0,
            fnr_noise = 0.0,
            protected_w = coefs_gen['protected'][i],
            score_w = coefs_gen['score'][i],
            seed = expert_seeds[i],
            theta = group_cfg['theta'],
            features_dict = group_cfg['w_dict']
        )
        #Creating the expert objects
        expert_team[expert_name] = experts.SigmoidExpert(**expert_args)
        expert_properties_list.append({**{'expert': expert_name}, **expert_args})
        EXPERT_IDS['human_ids'].append(expert_name)


#Fitting the experts
expert_team.fit(
    X=experts_train_X,
    y=val[LABEL_COL],
    score_col='model_score',
    protected_col=PROTECTED_COL,
)

#Saving expert's properties and parameters
full_w_table = pd.DataFrame(columns = experts_train_X.columns)
for expert in expert_team:
    full_w_table.loc[expert] = expert_team[expert].w

for expert in expert_team:
    full_w_table.loc[expert, 'fp_beta'] = expert_team[expert].fpr_beta
    full_w_table.loc[expert, 'fn_beta'] = expert_team[expert].fnr_beta
    full_w_table.loc[expert, 'alpha'] = expert_team[expert].alpha

os.makedirs(Path(__file__).parent/f'../../FiFAR/synthetic_experts/', exist_ok = True)
full_w_table.to_parquet(Path(__file__).parent/f'../../FiFAR/synthetic_experts/expert_parameters.parquet')

#Obtaining the predictions ----------------------------------------------------------------------------------

ml_train = val.copy()
ml_train[CATEGORICAL_COLS] = ml_train[CATEGORICAL_COLS].astype('category')

train_expert_pred = expert_team.predict(
    index=val.index,
    predict_kwargs={
        experts.SigmoidExpert: {
            'X': experts_train_X,
            'y': val[LABEL_COL]
        }}
)

deployment_expert_pred = expert_team.predict(
    index=train_test.index,
    predict_kwargs={
        experts.SigmoidExpert: {
            'X': experts_deployment_X,
            'y': train_test[LABEL_COL]
        }
        
    }
)

expert_pred = pd.concat([train_expert_pred,deployment_expert_pred])

expert_pred.to_parquet(Path(__file__).parent/f'../../FiFAR/synthetic_experts/expert_predictions.parquet')

#saving the probability of error associated with each instance
perror = pd.DataFrame()

for expert in expert_team:
    if(expert) != "model#0":
        column1 = f'p_fn_{expert}'
        column2 = f'p_fp_{expert}'
        perror[column1] = expert_team[expert].error_prob['p_of_fn']
        perror[column2] = expert_team[expert].error_prob['p_of_fp']


perror.to_parquet(Path(__file__).parent/f'../../FiFAR/synthetic_experts/prob_of_error.parquet')

with open(Path(__file__).parent/f'../../FiFAR/synthetic_experts/expert_ids.yaml', 'w') as outfile:
    yaml.dump(EXPERT_IDS, outfile)


print('Experts generated.')