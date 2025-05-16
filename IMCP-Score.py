from scoring_ts import loaddata_fromPDBbind_v2020, create_model, validate_model
import os
import numpy as np
import time
import itertools
import json
import warnings
from rdkit import RDLogger
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


from SIFt import SIFt

RDLogger.DisableLog('rdApp.*')

def cross_validate_models(X, y, n_splits=10, random_state=42):
    """
    Perform K-Fold CV for three regression models and return CV metrics.
    """
    models = {

        # 'RandomForest4': RandomForestRegressor(random_state=random_state,n_estimators = 300),
        # 'RandomForest5': RandomForestRegressor(random_state=random_state,n_estimators = 400),
        'RandomForest6': RandomForestRegressor(random_state=random_state,n_estimators = 500),
        'RandomForest7': RandomForestRegressor(random_state=random_state,n_estimators = 550),
        'RandomForest8': RandomForestRegressor(random_state=random_state,n_estimators = 600),
        'RandomForest9': RandomForestRegressor(random_state=random_state,n_estimators = 650),
        'RandomForest10': RandomForestRegressor(random_state=random_state,n_estimators = 700),
        # 'RandomForest11': RandomForestRegressor(random_state=random_state,n_estimators = 900),
        # 'RandomForest12': RandomForestRegressor(random_state=random_state,n_estimators = 950),  
        # 'GradientBoosting7': GradientBoostingRegressor(random_state=random_state,n_estimators=550),
        # 'GradientBoosting8': GradientBoostingRegressor(random_state=random_state,n_estimators=600),
        # 'GradientBoosting9': GradientBoostingRegressor(random_state=random_state,n_estimators=700),
        # 'GradientBoosting9': GradientBoostingRegressor(random_state=random_state,n_estimators=800),
        # 'GradientBoosting9': GradientBoostingRegressor(random_state=random_state,n_estimators=900),
        # "svr_poly1" : SVR(kernel='poly', degree=4, C=1.0, epsilon=0.1, coef0=1),
        # "svr_poly2" : SVR(kernel='poly', degree=5, C=1.0, epsilon=0.1, coef0=1),
        # "svr_poly3" : SVR(kernel='poly', degree=6, C=1.0, epsilon=0.1, coef0=1),
    }
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_results = {name: {'pearson': [], 'rmse': []} for name in models}
    # return cv_results, models
    for name, model in models.items():
        print("doing",name)
        for train_idx, val_idx in kf.split(X):
            print(len(val_idx))
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            # Metrics
            print(y_val.dtype, y_pred.dtype)
            r, _ = pearsonr(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            cv_results[name]['pearson'].append(r)
            cv_results[name]['rmse'].append(rmse)

        # Aggregate
        pearson_vals = cv_results[name]['pearson']
        rmse_vals    = cv_results[name]['rmse']

        pearson_mean = np.mean(pearson_vals)
        pearson_std  = np.std(pearson_vals)   # sample std
        rmse_mean    = np.mean(rmse_vals)
        rmse_std     = np.std(rmse_vals)      # sample std
        print(name)
        print(pearson_vals)
        print(rmse_vals)

        print(
            f"{name} CV -> "
            f"Pearson: {pearson_mean:.3f} ± {pearson_std:.3f}, "
            f"RMSE: {rmse_mean:.3f} ± {rmse_std:.3f}"
        )


    return cv_results, models


def train_and_test(X_train, y_train, X_test, y_test):
    # Load training data
    print(f"Training samples: {X_train.shape[0]}, features: {X_train.shape[1]}")

    # Cross-validate
    cv_results, models = cross_validate_models(X_train, y_train)

    # Train final models on all training data and evaluate on test set
    print(f"Test samples: {X_test.shape[0]}")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r, _ = pearsonr(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"{name} Test -> Pearson: {r:.3f}, RMSE: {rmse:.3f}")




###############################################################################################################
# run experiments                                                                                             #
###############################################################################################################

#############################################################################################################
# 1. generic scoring
#############################################################################################################
sd = 50
np.random.seed(sd)
data_dir = "/home/ansh-meshram/Desktop/work/bio_project/PDBbind"
spr_train = [0.9, 0.1, 0]
spr_test = [1, 0, 0]
trparadict = {'rf': {'rf_n_estimators': np.arange(300, 800, 100).tolist()},
              'gb': {'gb_n_estimators': np.arange(300, 800, 100).tolist()},
               'sv' : {}  
            }
datasets = ['rs-casf2016-csarhiq', 'casf2016', 'csarhiqS1', 'csarhiqS2', 'csarhiqS3']
# 1.1. featurize datasets -----------------------------------------------------------------------------------
#################################### IMCPs #######################################################################
ifptp = 'rfscore_ext'
cur_cutoff = 18
cur_bins = [(0,18)]
para = {'addH': True, 'sant': False,
        'cutoff': cur_cutoff, 'ifp_type': ifptp,
        'bins': cur_bins}                                         
feat_dt_train = loaddata_fromPDBbind_v2020(data_dir = data_dir, 
                                           subset = datasets[0], 
                                           select_target = None, 
                                           randsplit_ratio = spr_train,
                                           para = para,
                                           rand_seed = sd)
feat_dt_test1 = loaddata_fromPDBbind_v2020(data_dir = data_dir, 
                                           subset = datasets[1], 
                                           select_target = None, 
                                           randsplit_ratio = spr_test,
                                           para = para,
                                           rand_seed = sd)

# 1.2 Model parameterization and validation -----------------------------------------------------------------
# results saved in 'res.txt' in the data folder -------------------------------------------------------------
print(json.dumps(para), '\nValidation results: \n')
print(feat_dt_train[1].dtype)
train_and_test(feat_dt_train[0],feat_dt_train[1],feat_dt_test1[0],feat_dt_test1[1])


