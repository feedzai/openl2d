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


class NpEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that extends the default JSONEncoder to handle NumPy data types.

    This encoder converts NumPy integers and floats to Python built-in types, and NumPy
    arrays to Python lists, so they can be encoded as JSON strings. All other objects
    are handled by the default JSONEncoder.

    Attributes:
        None

    Methods:
        default(obj): Overrides the default method to handle NumPy data types.

    Usage:
        Use this encoder in conjunction with the json.dumps() function to encode objects
        that contain NumPy data types as JSON strings.

    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

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

        self.trial_count = 0 

        self.train_w = train_w
        self.val_w = val_w

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.method = method
        self.parameters = parameters
        self.params_hist = []
        self.path = path
        self.current_best = +np.inf
        print(f"{self.path}/best_model.pickle", "wb")

    def initialize_optimizer(self, categorical, n_jobs):
        self.catcols = categorical
        self.tpe_optimization(n_jobs)
    
    def tpe_optimization(self, n_jobs):

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed = 42, n_startup_trials=100)
        )

        study.set_user_attr("X_train", self.X_train)
        study.set_user_attr("X_val", self.X_val)
        study.set_user_attr("y_train", self.y_train)
        study.set_user_attr("y_val", self.y_val)
        study.set_user_attr("n_jobs", n_jobs)

        optuna.logging.set_verbosity(optuna.logging.FATAL)
        study.optimize(self.objective, n_trials=120)

        trials_df = study.trials_dataframe()
        keys = trials_df.keys()

        with open(
            "{}/config.yaml".format(self.path), "w"
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
        max_depth = trial.suggest_int("max_depth",2,20,log = False)
        n_estimators = trial.suggest_int("n_estimators", 50, 250, log=False)
        num_leaves = trial.suggest_int("num_leaves", 100, 1000, log=True)
        min_child_samples = trial.suggest_int("min_child_samples", 5,100, log=True)
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.5, log=True)
        reg_alpha = trial.suggest_float("reg_alpha", 0.0001, 0.1, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 0.0001, 0.1, log=True)

        param_dict = {
            "enable_bundle" : enable_bundle,
            "max_depth": max_depth,
            "n_estimators" : n_estimators,
            "num_leaves" :num_leaves,
            "min_child_samples" : min_child_samples,
            "learning_rate" : learning_rate,
            "reg_alpha" :reg_alpha,
            "reg_lambda" : reg_lambda
        }

        self.params_hist.append(param_dict)

        # Train the model with the given hyperparameters
        model = lgb.LGBMClassifier(
            importance_type='gain',
            max_depth = max_depth,
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

        print(model)

        print(f'Fitting model {self.trial_count}')
        model.fit(
            X_train,
            y_train,
            sample_weight= self.train_w,
            verbose=False
        )
        self.trial_count +=1
        # Evaluate the model on the testing data
        y_pred = model.predict_proba(X_val)
        y_pred = y_pred[:, 1]

        # GET THRESHOLD at 5% fpr
        results = pd.DataFrame()
        results["true"] = y_val
        results["score"] = y_pred

        ll = log_loss(y_true = y_val, y_pred = y_pred, sample_weight=self.val_w)

        if ll < self.current_best:
            self.current_best = ll
            os.makedirs(self.path, exist_ok = True)
            with open(f"{self.path}/best_model.pickle", "wb") as fout:
                pickle.dump(model, fout)

        return ll  
    