# %%
import pandas as pd
import yaml
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer, StandardScaler
import pickle
from datetime import date
from autodefer.models import haic
import numpy as np
import os
from pathlib import Path



cfg_path = Path(__file__).parent/'./cfg.yaml'
with open(cfg_path, 'r') as infile:
    cfg = yaml.safe_load(infile)

data_cfg_path = Path(__file__).parent/'../data/dataset_cfg.yaml'
with open(data_cfg_path, 'r') as infile:
    data_cfg = yaml.safe_load(infile)

cat_dict = data_cfg['categorical_dict']

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            print(f'{feature} has been reencoded')
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data

data = pd.read_parquet(Path(__file__).parent/'../data/BAF.parquet')

LABEL_COL = data_cfg['data_cols']['label']
TIMESTAMP_COL = data_cfg['data_cols']['timestamp']
PROTECTED_COL = data_cfg['data_cols']['protected']
CATEGORICAL_COLS = data_cfg['data_cols']['categorical']

data = cat_checker(data, CATEGORICAL_COLS, cat_dict)

def splitter(df, timestamp_col, beginning: int, end: int):
    return df[
        (df[timestamp_col] >= beginning) &
        (df[timestamp_col] < end)].copy()

train = splitter(data, TIMESTAMP_COL, *cfg['splits']['train']).drop(columns=TIMESTAMP_COL)
ml_val = splitter(data, TIMESTAMP_COL, *cfg['splits']['ml_val']).drop(columns=TIMESTAMP_COL)
deployment = splitter(data, TIMESTAMP_COL, *cfg['splits']['deployment']).drop(columns=TIMESTAMP_COL)

# EXPERTS ------------------------------------------------------------------------------------------
expert_team = haic.experts.ExpertTeam()
EXPERT_IDS = dict(model_ids=list(), human_ids=list())
THRESHOLDS = dict()

with open(Path(__file__).parent/'../ml_model/model/best_model.pickle', 'rb') as infile:
    ml_model = pickle.load(infile)

with open(Path(__file__).parent/'../ml_model/model/model_properties.pickle', 'rb') as infile:
    ml_model_properties = pickle.load(infile)

ml_model_threshold = ml_model_properties['threshold']
ml_model_recall = 1 - ml_model_properties['fnr']
ml_model_fpr_diff = ml_model_properties['disparity']
ml_model_fpr = ml_model_properties['fpr']

expert_team['model#0'] = haic.experts.MLModelExpert(fitted_model=ml_model, threshold=None)
EXPERT_IDS['model_ids'].append('model#0')
THRESHOLDS['model#0'] = ml_model_threshold


if( os.path.isfile(Path(__file__).parent/'./transformed_data/X_deployment_experts.parquet') and os.path.isfile(Path(__file__).parent/'./transformed_data/X_deployment_experts.parquet')):
    experts_deployment_X = pd.read_parquet(Path(__file__).parent/'./transformed_data/X_deployment_experts.parquet')
    experts_train_X = pd.read_parquet(Path(__file__).parent/'./transformed_data/X_train_experts.parquet')
else:
    experts_train_X = train.copy().drop(columns=LABEL_COL)
    experts_train_X['score'] = expert_team[EXPERT_IDS['model_ids'][0]].predict(
        train.drop(columns=LABEL_COL))
    experts_train_X[PROTECTED_COL] = (experts_train_X[PROTECTED_COL] >= 50).astype(int)

    experts_deployment_X = deployment.copy().drop(columns=LABEL_COL)
    experts_deployment_X['score'] = expert_team[EXPERT_IDS['model_ids'][0]].predict(
        deployment.drop(columns=LABEL_COL))
    experts_deployment_X[PROTECTED_COL] = (experts_deployment_X[PROTECTED_COL] >= 50).astype(int)

    new_score_train = pd.Series(index = experts_train_X.index, dtype = 'float64')
    new_score_deployment = pd.Series(index = experts_deployment_X.index, dtype = 'float64')

    new_score_train.loc[experts_train_X['score'] <= ml_model_threshold] = (0.5/ml_model_threshold) * experts_train_X['score'].loc[experts_train_X['score'] <= ml_model_threshold] - 0.5
    new_score_train.loc[experts_train_X['score'] > ml_model_threshold] = (0.5/(1 - ml_model_threshold)) * experts_train_X['score'].loc[experts_train_X['score'] > ml_model_threshold] + (0.5 - (0.5/(1-ml_model_threshold)))

    new_score_deployment.loc[experts_deployment_X['score'] <= ml_model_threshold] = (0.5/ml_model_threshold) * experts_deployment_X['score'].loc[experts_deployment_X['score'] <= ml_model_threshold] - 0.5
    new_score_deployment.loc[experts_deployment_X['score'] > ml_model_threshold] = (0.5/(1 - ml_model_threshold)) * experts_deployment_X['score'].loc[experts_deployment_X['score'] > ml_model_threshold] + (0.5 - (0.5/(1-ml_model_threshold)))

    cols_to_quantile = experts_train_X.drop(columns=CATEGORICAL_COLS).columns.tolist()
    qt = QuantileTransformer(random_state=42)
    experts_train_X[cols_to_quantile] = (
        qt.fit_transform(experts_train_X[cols_to_quantile])
        - 0.5  # centered on 0
    )

    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    experts_train_X[CATEGORICAL_COLS] = oe.fit_transform(experts_train_X[CATEGORICAL_COLS])

    #Payment type for example, is now converted.

    ss = StandardScaler(with_std=False)
    experts_train_X[:] = ss.fit_transform(experts_train_X)

    cols_to_scale = [c for c in experts_train_X.columns if c not in cols_to_quantile]
    desired_range = 1
    scaling_factors = (
        desired_range /
        (experts_train_X[cols_to_scale].max() - experts_train_X[cols_to_scale].min())
    )
    experts_train_X[cols_to_scale] *= scaling_factors

    # preprocess other splits
    def preprocess(df):
        processed_X = df.copy()
        processed_X[cols_to_quantile] = qt.transform(processed_X[cols_to_quantile]) - 0.5  # centered on 0
        processed_X[CATEGORICAL_COLS] = oe.transform(processed_X[CATEGORICAL_COLS])
        processed_X[:] = ss.transform(processed_X)
        processed_X[cols_to_scale] *= scaling_factors

        return processed_X


    experts_deployment_X = preprocess(experts_deployment_X)
    experts_train_X['score'] = new_score_train
    experts_deployment_X['score'] = new_score_deployment

    experts_deployment_X.to_parquet(Path(__file__).parent/'./transformed_data/X_deployment_experts.parquet')
    experts_train_X.to_parquet(Path(__file__).parent/'./transformed_data/X_train_experts.parquet')

# %%
# 2.2 GENERATION -----------------------------------------------------------------------------------
if os.path.isdir(Path(__file__).parent/'./expert_info/'):
    print('Experts already Generated')
else:
    def process_groups_cfg(groups_cfg, baseline_name='regular'):
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
    for group_name, group_cfg in ensemble_cfg.items():
        np.random.seed(group_cfg['group_seed'])
        # substitute anchored values by actual values
        if group_cfg['fnr']['intercept_mean'] == 'model - stdev':
            group_cfg['fnr']['intercept_mean'] = (
                (1 - ml_model_recall)
                - group_cfg['fnr']['intercept_stdev']
            )
        if group_cfg['fpr']['intercept_mean'] == 'model - stdev':
            group_cfg['fpr']['intercept_mean'] = (
                ml_model_fpr
                - group_cfg['fpr']['intercept_stdev']
            )

        coefs_gen = dict()
        for coef in ['score', 'protected', 'alpha']:
            coefs_gen[coef] = np.random.normal(
                    loc=group_cfg[f'{coef}_mean'],
                    scale=group_cfg[f'{coef}_stdev'],
                    size=group_cfg['n']
            )
        
        coefs_spe = dict()
        for eq in ['fnr', 'fpr']:
            coefs_spe[eq] = dict()
            for coef in ['intercept']:
                coefs_spe[eq][coef] = np.random.normal(
                    loc=group_cfg[eq][f'{coef}_mean'],
                    scale=group_cfg[eq][f'{coef}_stdev'],
                    size=group_cfg['n']
            )
                
        expert_seeds = np.random.randint(low = 2**32-1, size = group_cfg['n'])

        for i in range(group_cfg['n']):
            expert_name = f'{group_name}#{i}'
            expert_args = dict(
                fnr_base=coefs_spe['fnr']['intercept'][i],
                fpr_base=coefs_spe['fpr']['intercept'][i], #Purposefully setting the same one
                features_w_std = group_cfg['w_std'],
                alpha = coefs_gen['alpha'][i],
                fpr_noise = 0.0,
                fnr_noise = 0.0,
                protected_w = coefs_gen['protected'][i],
                score_w = coefs_gen['score'][i],
                seed = expert_seeds[i]
            )
            expert_team[expert_name] = haic.experts.SigmoidExpert(**expert_args)
            expert_properties_list.append({**{'expert': expert_name}, **expert_args})
            EXPERT_IDS['human_ids'].append(expert_name)


    expert_team.fit(
        X=experts_train_X,
        y=train[LABEL_COL],
        score_col='score',
        protected_col=PROTECTED_COL,
    )

    full_w_table = pd.DataFrame(columns = experts_train_X.columns)
    for expert in expert_team:
        if(expert) != "model#0":
            full_w_table.loc[expert] = expert_team[expert].w

    for expert in expert_team:
        if(expert) != "model#0":
            full_w_table.loc[expert, 'fp_intercept'] = expert_team[expert].fpr_intercept
            full_w_table.loc[expert, 'fn_intercept'] = expert_team[expert].fnr_intercept
            full_w_table.loc[expert, 'alpha'] = expert_team[expert].alpha

    os.makedirs(Path(__file__).parent/'./expert_info', exist_ok = True)
    full_w_table.to_parquet(Path(__file__).parent/'./expert_info/full_w_table.parquet')

    # properties
    expert_properties = pd.DataFrame(expert_properties_list)

    expert_properties.to_parquet(Path(__file__).parent/'./expert_info/expert_properties.parquet')

    # 2.2 PREDICTIONS ----------------------------------------------------------------------------------
    #Leo: This is the file with all the expert predictions for training
    #Leo: These contain all the expert's 0,1 predictions.
    ml_train = train.copy()
    ml_train[CATEGORICAL_COLS] = ml_train[CATEGORICAL_COLS].astype('category')

    train_expert_pred = expert_team.predict(
        index=train.index,
        predict_kwargs={
            haic.experts.SigmoidExpert: {
                'X': experts_train_X,
                'y': train[LABEL_COL]
            },
            haic.experts.MLModelExpert: {
                'X': ml_train.drop(columns=[LABEL_COL])
            }}
    )

    train_expert_pred.to_parquet(Path(__file__).parent/'./expert_info/train_predictions.parquet')

    deployment_expert_pred = expert_team.predict(
        index=deployment.index,
        predict_kwargs={
            haic.experts.SigmoidExpert: {
                'X': experts_deployment_X,
                'y': deployment[LABEL_COL]
            },
            haic.experts.MLModelExpert: {
                'X': deployment.drop(columns=[LABEL_COL])
            }
        },
    )
    deployment_expert_pred.to_parquet(Path(__file__).parent/'./expert_info/deployment_predictions.parquet')

    perror = pd.DataFrame()

    for expert in expert_team:
        if(expert) != "model#0":
            column1 = f'p_fn_{expert}'
            column2 = f'p_fp_{expert}'
            perror[column1] = expert_team[expert].error_prob['p_of_fn']
            perror[column2] = expert_team[expert].error_prob['p_of_fp']


    perror.to_parquet(Path(__file__).parent/'./expert_info/p_of_error.parquet')



    # %%
    with open(Path(__file__).parent/'./expert_info/expert_ids.yaml', 'w') as outfile:
        yaml.dump(EXPERT_IDS, outfile)


    print('Experts generated.')