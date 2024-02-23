import json
import pickle

import numpy as np
import optuna
import pandas as pd
from scipy.stats import loguniform
from sklearn.metrics import recall_score, log_loss
import lightgbm as lgb
import os
import yaml

def sig(x):
    return 1/(1+np.exp(-x))

def output(data, model, init_score):
    return sig(model.predict(data,raw_score=True) + init_score)



class HPO:

    def __init__(
        self,
        X_train,
        X_val,
        y_train,
        y_val,
        train_w,
        val_w,
        method="TPE",
        parameters=None,
        path="",):

        parameters['init_score'] = float(parameters['init_score'])
        self.max_trials = int(parameters['total'])
        self.startup = int(parameters['startups'])
        self.trial_count = 0 

        self.train_w = train_w
        self.val_w = val_w

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        if parameters['init_score'] == 'None':
            self.init_score = None
        else:
            self.init_score = float(parameters['init_score'])*np.ones(len(X_train))
        self.method = method
        self.config = parameters
        self.parameters = parameters['params']
        self.params_hist = []
        self.path = path
        self.current_best = +np.inf

        

    def initialize_optimizer(self, categorical, n_jobs):
        self.catcols = categorical
        self.tpe_optimization(n_jobs)
    
    def tpe_optimization(self, n_jobs):

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed = 42, n_startup_trials=self.startup)
        )

        study.set_user_attr("X_train", self.X_train)
        study.set_user_attr("X_val", self.X_val)
        study.set_user_attr("y_train", self.y_train)
        study.set_user_attr("y_val", self.y_val)
        study.set_user_attr("n_jobs", n_jobs)

        os.makedirs(self.path, exist_ok = True)
        with open(
            "{}/config.yaml".format(self.path), "w"
        ) as fout:
            yaml.dump(self.config, fout)

        optuna.logging.set_verbosity(optuna.logging.FATAL)
        study.optimize(self.objective, n_trials=self.max_trials)

        trials_df = study.trials_dataframe()
        keys = trials_df.keys()

        with open(
            "{}/history.yaml".format(self.path), "w"
        ) as fout:
            yaml.dump(self.params_hist, fout)
        
        
        return
        


    def objective(self, trial):  # Instance method
        """Optuna objective function for LightGBM model hyperparameter optimization.

        Args:
            trial: An Optuna trial object.

        Returns:
            The true positive rate (recall score) on the validation set, to maximize.
        """

        X_train = trial.study.user_attrs["X_train"]
        X_val = trial.study.user_attrs["X_val"]
        y_train = trial.study.user_attrs["y_train"]
        y_val = trial.study.user_attrs["y_val"]
        n_jobs = trial.study.user_attrs["n_jobs"]
        
        # Define the hyperparameters to optimize
        #boosting_type = trial.suggest_categorical("boosting_type", ["dart"])

        enable_bundle = trial.suggest_categorical("enable_bundle", [True, False])


        max_depth = trial.suggest_int("max_depth", self.parameters['max_depth']['range'][0], self.parameters['max_depth']['range'][1], log = self.parameters['max_depth']['log'])
        n_estimators = trial.suggest_int("n_estimators", self.parameters['n_estimators']['range'][0], self.parameters['n_estimators']['range'][1], log = self.parameters['n_estimators']['log'])
        num_leaves = trial.suggest_int("num_leaves", self.parameters['num_leaves']['range'][0], self.parameters['num_leaves']['range'][1], log = self.parameters['num_leaves']['log'])
        min_child_samples = trial.suggest_int("min_child_samples", self.parameters['min_child_samples']['range'][0], self.parameters['min_child_samples']['range'][1],log = self.parameters['min_child_samples']['log'])
        learning_rate = trial.suggest_float("learning_rate", self.parameters['learning_rate']['range'][0], self.parameters['learning_rate']['range'][1],log = self.parameters['learning_rate']['log'])
        reg_alpha = trial.suggest_float("reg_alpha", self.parameters['reg_alpha']['range'][0], self.parameters['reg_alpha']['range'][1], log = self.parameters['reg_alpha']['log'])
        reg_lambda = trial.suggest_float("reg_lambda", self.parameters['reg_lambda']['range'][0], self.parameters['reg_lambda']['range'][1], log = self.parameters['reg_lambda']['log'])

        
        # Train the model with the given hyperparameters
        model = lgb.LGBMClassifier(
            importance_type='gain',
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_samples=min_child_samples,
            num_leaves=num_leaves,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            boosting_type="dart",
            enable_bundle=enable_bundle,
            seed = 42
        )

        print(f'Fitting model {self.trial_count}')
        model.fit(
            X_train,
            y_train,
            sample_weight= self.train_w,
            init_score = self.init_score,
            verbose=False
        )
        self.trial_count +=1
        # Evaluate the model on the testing data
        if self.init_score is None:
            y_pred = model.predict_proba(X_val)[:,1]
        else:
            y_pred = output(X_val,model,self.init_score[0])

        ll = log_loss(y_true = y_val, y_pred = y_pred, sample_weight=self.val_w)

        if self.init_score is None:
            param_dict = {
            "init_score": 'None',
            "max_depth": max_depth,
            "N_trial": self.trial_count,
            "enable_bundle" : enable_bundle,
            "n_estimators" : n_estimators,
            "num_leaves" :num_leaves,
            "min_child_samples" : min_child_samples,
            "learning_rate" : learning_rate,
            "reg_alpha" :reg_alpha,
            "reg_lambda" : reg_lambda,
            'll': float(ll)
        }
        else:
            param_dict = {
                "init_score": float(self.init_score[0]),
                "max_depth": max_depth,
                "N_trial": self.trial_count,
                "enable_bundle" : enable_bundle,
                "n_estimators" : n_estimators,
                "num_leaves" :num_leaves,
                "min_child_samples" : min_child_samples,
                "learning_rate" : learning_rate,
                "reg_alpha" :reg_alpha,
                "reg_lambda" : reg_lambda,
                'll': float(ll)
            }

        self.params_hist.append(param_dict)


        if ll < self.current_best:
            self.current_best = ll
            os.makedirs(self.path, exist_ok = True)
            with open(f"{self.path}/best_model.pickle", "wb") as fout:
                pickle.dump(model, fout)
            with open("{}/best_config.yaml".format(self.path), "w") as fout:
                yaml.dump([param_dict], fout)

        return ll  