# %%
import os
import random

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from autodefer.models import haic

cfg_path = 'cfg.yaml'
with open(Path(__file__).parent/ cfg_path, 'r') as infile:
    cfg = yaml.safe_load(infile)

with open(Path(__file__).parent/ cfg['data_cfg_path'], 'r') as infile:
    data_cfg = yaml.safe_load(infile)

with open(Path(__file__).parent/ cfg['expert_ids_path'], 'r') as infile:
    EXPERT_IDS = yaml.safe_load(infile)

random.seed(cfg['random_seed'])
np_rng = np.random.default_rng(cfg['random_seed'])


# DATA LOADING -------------------------------------------------------------------------------------
data = pd.read_parquet(Path(__file__).parent/ '../data/BAF.parquet')
TIMESTAMP_COL = data_cfg['data_cols']['timestamp']
EXPERT_CATS = EXPERT_IDS['human_ids'] +  EXPERT_IDS['model_ids']

def splitter(df, timestamp_col, beginning: int, end: int):
    return df[
        (df[timestamp_col] >= beginning) &
        (df[timestamp_col] < end)].copy()

train = splitter(data, TIMESTAMP_COL, *cfg['splits']['train'])
test = splitter(data, TIMESTAMP_COL, *cfg['splits']['test'])

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data



cat_dict = data_cfg['categorical_dict']

train = cat_checker(train, data_cfg['data_cols']['categorical'], cat_dict)
test = cat_checker(test, data_cfg['data_cols']['categorical'], cat_dict)

# EXPERTS ------------------------------------------------------------------------------------------
# produced in experts/experts_generation.py
experts_pred = pd.read_parquet(Path(__file__).parent/cfg['experts_pred_path'])
train_expert_pred = experts_pred.loc[train.index, ]
test_expert_pred = experts_pred.loc[test.index, ]

# BATCH & CAPACITY ---------------------------------------------------------------------------------
def generate_batches(df, batch_properties: dict, months: pd.Series) -> pd.DataFrame:
    """
    Generates a pandas dataframe indicating the (serial) number of the batch each instance belongs to.
    Batches do not crossover from one month to the other.
    :param batch_properties: dictionary containing size key-value pair (see cfg.yaml).
    :param months: pandas series indicating the month of each instance.
    """
    batches_months_list = list()
    last_batch_ix = 0
    for m in months.unique():
        df_m = df[months == m]
        df_m = df_m.sample(frac = 1, random_state = batch_properties['seed'])
        
        m_batches = pd.DataFrame(
            [int(i / batch_properties['size']) + last_batch_ix + 1 for i in range(len(df_m))],
            index=df_m.index,
            columns=['batch'],
        )
        batches_months_list.append(m_batches)
        last_batch_ix = int(m_batches.max())

    batches = pd.concat(batches_months_list)

    return batches

def generate_capacity_single_batch(batch_size: int, properties: dict, model_id: str, human_ids: list) -> dict:
    """
    Generates dictionary indicating the capacity of each decision-maker (from model_id and human_ids).
    This capacity pertains to a single batch.
    :param properties: dictionary indicating capacity constraints (see cfg.yaml)
    :param model_id: identification of the model to be used in the output dictionary.
    :param human_ids: identification of the humans to be used in the output dictionary.
    """
    capacity_dict = dict()
    capacity_dict['batch_size'] = batch_size
    capacity_dict[model_id] = int((1 - properties['deferral_rate']) * batch_size)
    if properties['distribution'] == 'homogeneous':
        humans_capacity_value = int(
            (batch_size - capacity_dict[model_id]) /
            len(human_ids)
        )
        unc_human_capacities = np.full(shape=(len(human_ids),), fill_value=humans_capacity_value)
    elif properties['distribution'] == 'variable':  # capacity follows a random Gaussian
        mean_individual_capacity = (batch_size - capacity_dict[model_id]) / len(human_ids)
        np_rng = np.random.default_rng(properties['distribution_seed'])
        unc_human_capacities = np_rng.normal(
            loc=mean_individual_capacity,
            scale=properties['distribution_stdev'] * mean_individual_capacity,
            size=(len(human_ids),),
        )
        unc_human_capacities += (
            (batch_size - capacity_dict[model_id] - sum(unc_human_capacities))
            / len(human_ids)
        )

    if 'pool' in properties:
        available_humans_ix = []
        absent_humans_ix = []
        for human_ix, human in enumerate(human_ids):
            if human.split('#')[0] in properties['pool']:
                available_humans_ix.append(human_ix)
            else:
                absent_humans_ix.append(human_ix)
        unc_human_capacities[absent_humans_ix] = 0
        unc_human_capacities = unc_human_capacities.astype(float)
        unassigned = (batch_size - capacity_dict[model_id] - sum(unc_human_capacities))
        unc_human_capacities[available_humans_ix] *= (1 + unassigned / sum(unc_human_capacities))
    else:
        absent_humans_ix = []
        available_humans_ix = list(range(len(human_ids)))

    if 'absence' in properties:  # some experts are randomly unavailable
        random.seed(properties['absence_seed'])
        absent_humans_ix += random.sample(  # without replacement
            available_humans_ix,
            k=int(properties['absence'] * len(available_humans_ix)),
        )
        unc_human_capacities[absent_humans_ix] = 0

        unassigned = (batch_size - capacity_dict[model_id] - sum(unc_human_capacities))
        available_humans_ix = [ix for ix in available_humans_ix if ix not in absent_humans_ix]

        unc_human_capacities = unc_human_capacities.astype(float)
        unc_human_capacities[available_humans_ix] *= (1 + unassigned / sum(unc_human_capacities))

    # convert to integer and adjust for rounding errors
    human_capacities = np.floor(unc_human_capacities).astype(int)
    unassigned = int(batch_size - capacity_dict[model_id] - sum(human_capacities))
    
    assert unassigned < len(human_ids)
    random.seed(42)
    to_add_to = random.sample(available_humans_ix, k=unassigned)
    human_capacities[to_add_to] += 1

    capacity_dict.update(**{
        human_ids[ix]: int(human_capacities[ix])
        for ix in range(len(human_ids))
    })

    assert sum(list(capacity_dict.values())[1:]) == list(capacity_dict.values())[0]

    return capacity_dict

def generate_capacity(batches: pd.Series, capacity_properties: dict) -> pd.DataFrame:
    """
    Generates pandas dataframe matching batch_ids to capacity constraints for that batch.
    :param batches: pandas dataframe output by generate_batches()
    :param capacity_properties: dictionary output by generate_capacity_single_batch()
    """
    if 'batch_shuffle' in capacity_properties:
        dictionary = dict()
        n_batches = len(batches.iloc[:, 0].unique())
        absent_seeds = None
        distribution_seeds = None
        if capacity_properties['distribution'] == 'variable' and 'distribution' in capacity_properties['batch_shuffle']:
            np.random.seed(capacity_properties['distribution_seed'])
            distribution_seeds = np.random.randint(low = 1e10, size = n_batches)
        if 'absence' in capacity_properties and 'absence' in capacity_properties['batch_shuffle']:
            np.random.seed(capacity_properties['absence_seed'])
            absent_seeds = np.random.randint(low = 1e10, size = n_batches)

        for b_ix in batches.iloc[:, 0].unique():
            if absent_seeds is not None:
                capacity_properties['absence_seed'] = absent_seeds[b_ix-1]
            if distribution_seeds is not None:
                capacity_properties['distribution_seed'] = distribution_seeds[b_ix-1]
            dictionary[b_ix] = generate_capacity_single_batch(
                batch_size=int((batches == b_ix).sum()),
                properties=capacity_properties,
                model_id=EXPERT_IDS['model_ids'][0],
                human_ids=EXPERT_IDS['human_ids'],
                )
        
        capacity_df = pd.DataFrame.from_dict(dictionary).T

    else:
        capacity_df = pd.DataFrame.from_dict(
            {
                int(b_ix): generate_capacity_single_batch(
                    batch_size=int((batches == b_ix).sum()),
                    properties=capacity_properties,
                    model_id=EXPERT_IDS['model_ids'][0],
                    human_ids=EXPERT_IDS['human_ids'],
                )
                for b_ix in batches.iloc[:, 0].unique()
            },
            orient='index'
        )

    return capacity_df.drop(columns = 'model#0')

def generate_environments(df, batch_cfg: dict, capacity_cfg: dict, output_dir=None) -> dict:
    """
    Generates a dictionary matching environment keys to batch and capacity dataframes.
    :param batch_cfg: dictionary with the batch configurations (see cfg.yaml).
    :param capacity_cfg: dictionary with the capacity configurations (see cfg.yaml).
    :param output_dir: directory to save to.
    """
    environments = dict()
    for batch_scheme, batch_properties in batch_cfg.items():
        for capacity_scheme, capacity_properties in capacity_cfg.items():
            batches_df = generate_batches(
                df=df,
                batch_properties=batch_properties,
                months=df[TIMESTAMP_COL]
            )
            capacity_df = generate_capacity(
                batches=batches_df, capacity_properties=capacity_properties)
            if output_dir is not None:
                env_path = f'{output_dir}{batch_scheme}#{capacity_scheme}/'
                os.makedirs(Path(__file__).parent/env_path, exist_ok=True)
                batches_df.to_csv(Path(__file__).parent/ env_path /'batches.csv')
                capacity_df.index.names = ['batch_id']
                capacity_df.to_csv(Path(__file__).parent/ env_path / 'capacity.csv')
            environments[(batch_scheme, capacity_scheme)] = (batches_df, capacity_df)

    return environments


# METADATA -----------------------------------------------------------------------------------------
BATCH_COL = 'batch'
ASSIGNMENT_COL = 'assignment'
DECISION_COL = 'decision'

train_metadata = {
    'expert_ids': EXPERT_IDS,
    'data_cols': {
        **data_cfg['data_cols'],
        'score': 'model_score',
        'batch': BATCH_COL,
        'assignment': ASSIGNMENT_COL,
        'decision': DECISION_COL,
    }
}
with open(Path(__file__).parent/ cfg['output_paths']['metadata'], 'w') as outfile:
    yaml.dump(train_metadata, outfile)


# TEST ---------------------------------------------------------------------------------------------
os.makedirs(Path(__file__).parent/ cfg['output_paths']['test_dir'], exist_ok=True)
test = test.assign(
    model_score=test_expert_pred[EXPERT_IDS['model_ids'][0]]
)
test.index.name = 'case_id'
test = test.rename(columns = {'fraud_bool': 'fraud_label'})
test.to_parquet(Path(__file__).parent/ (cfg['output_paths']['test_dir'] + 'test.parquet'))
test_expert_pred.index.name = 'case_id'
test_expert_pred.to_parquet(Path(__file__).parent / (cfg['output_paths']['test_dir'] + 'test_expert_pred.parquet'))

generate_environments(
    df=test,
    batch_cfg=cfg['environments_test']['batch'],
    capacity_cfg=cfg['environments_test']['capacity'],
    output_dir=cfg['output_paths']['test_dir']
)

print('Testbed generated.')