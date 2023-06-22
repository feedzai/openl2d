# %%
import os
import random

import numpy as np
import pandas as pd
import yaml

from autodefer.models import haic

cfg_path = 'cfg.yaml'
with open(cfg_path, 'r') as infile:
    cfg = yaml.safe_load(infile)

with open(cfg['data_cfg_path'], 'r') as infile:
    data_cfg = yaml.safe_load(infile)

with open(cfg['expert_ids_path'], 'r') as infile:
    EXPERT_IDS = yaml.safe_load(infile)

random.seed(cfg['random_seed'])
np_rng = np.random.default_rng(cfg['random_seed'])

# DATA LOADING -------------------------------------------------------------------------------------
data = pd.read_parquet('../data/BAF.parquet')
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
experts_pred = pd.read_parquet(cfg['experts_pred_path'])
##All expert predictions here.
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
    capacity_dict[model_id] = int((1 - properties['deferral_rate']) * batch_size)

    if properties['distribution'] == 'homogeneous':
        humans_capacity_value = int(
            (batch_size - capacity_dict[model_id]) /
            len(human_ids)
        )
        unc_human_capacities = np.full(shape=(len(human_ids),), fill_value=humans_capacity_value)
    elif properties['distribution'] == 'variable':  # capacity follows a random Gaussian
        mean_individual_capacity = (batch_size - capacity_dict[model_id]) / len(human_ids)
        unc_human_capacities = np_rng.normal(
            loc=mean_individual_capacity,
            scale=properties['distribution_stdev'] * mean_individual_capacity,
            size=(len(human_ids),),
        )
        unc_human_capacities += (
            (batch_size - capacity_dict[model_id] - sum(unc_human_capacities))
            / len(human_ids)
        )

    available_humans_ix = list(range(len(human_ids)))
    if 'absence' in properties:  # some experts are randomly unavailable
        absent_humans_ix = random.sample(  # without replacement
            available_humans_ix,
            k=int(properties['absence'] * len(human_ids)),
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
    to_add_to = random.sample(available_humans_ix, k=unassigned)
    human_capacities[to_add_to] += 1

    capacity_dict.update(**{
        human_ids[ix]: int(human_capacities[ix])
        for ix in range(len(human_ids))
    })

    assert sum(capacity_dict.values()) == batch_size

    return capacity_dict

def generate_capacity(batches: pd.Series, capacity_properties: dict) -> pd.DataFrame:
    """
    Generates pandas dataframe matching batch_ids to capacity constraints for that batch.
    :param batches: pandas dataframe output by generate_batches()
    :param capacity_properties: dictionary output by generate_capacity_single_batch()
    """
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
    return capacity_df

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
                os.makedirs(env_path, exist_ok=True)
                batches_df.to_parquet(env_path+'batches.parquet')
                capacity_df.to_parquet(env_path+'capacity.parquet')
            environments[(batch_scheme, capacity_scheme)] = (batches_df, capacity_df)

    return environments

def generate_predictions(X, expert_pred, batches, capacity, output_dir=None):
    """
    Randomly assigns instances to decision-makers. Queries said decision-makers for decisions.
    Returns X dataframe merged with assignments and decisions.
    :param X: full dataset, including features.
    :param expert_pred: full matrix of expert predictions for X.
    :param batches: output of generate_batches().
    :param capacity: output of generate_capacity().
    :param output_dir: directory to save to.
    """
    assgn_n_dec = X.merge(batches, left_index=True, right_index=True)

    random_assigner = haic.assigners.RandomAssigner(expert_ids=EXPERT_IDS)
    assgn_n_dec[ASSIGNMENT_COL] = random_assigner.assign(
        assgn_n_dec,
        batch_col=batches.columns[0],
        capacity=capacity.T.to_dict(),
    )
    assgn_n_dec[ASSIGNMENT_COL] = pd.Categorical(assgn_n_dec[ASSIGNMENT_COL].values, categories = EXPERT_CATS)
    assgn_n_dec[DECISION_COL] = haic.query_experts(
        pred=expert_pred, assignments=assgn_n_dec[ASSIGNMENT_COL])
    assgn_n_dec = assgn_n_dec[[ASSIGNMENT_COL, DECISION_COL]]

    if output_dir is not None:
        assgn_n_dec.to_parquet(output_dir + 'assignments_and_decisions.parquet')

    return assgn_n_dec

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
with open(cfg['output_paths']['metadata'], 'w') as outfile:
    yaml.dump(train_metadata, outfile)

# TRAIN --------------------------------------------------------------------------------------------
os.makedirs(cfg['output_paths']['train_dir'], exist_ok=True)

train_envs = generate_environments(
    df=train,
    batch_cfg=cfg['environments_train']['batch'],
    capacity_cfg=cfg['environments_train']['capacity'],
    output_dir=cfg['output_paths']['train_dir'],
)
for (batch_scheme, capacity_scheme), (train_batches, train_capacity) in train_envs.items():
    env_assignment_and_pred = generate_predictions(
        X=train,
        expert_pred=train_expert_pred,
        batches=train_batches,
        capacity=train_capacity,
    )
    env_train = (
        train
        .assign(model_score=train_expert_pred[EXPERT_IDS['model_ids'][0]])
        .merge(train_batches, left_index=True, right_index=True)
        .merge(env_assignment_and_pred, left_index=True, right_index=True)
    )
    env_train.to_parquet(
        f"{cfg['output_paths']['train_dir']}{batch_scheme}#{capacity_scheme}/train.parquet"
    )