import numpy as np
import pandas as pd 
import os 
import shutil
import time 
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from metric_learn import MLKR
import matplotlib.pyplot as plt 

class Emb():
    def __init__(self, data_df, nlfs_result, targ_var):
        self.data_df = data_df
        pred_vars = nlfs_result.loc[np.argmin(nlfs_result["best_mae"].values), "label"].split("|")
        self.pred_vars = pred_vars
        self.X = data_df[pred_vars].values
        self.targ_var = targ_var
        self.y = data_df[targ_var].values
    
    def fit(self, random_state=1, return_coords=False):
        self.feature_scaler = MinMaxScaler().fit(self.X)
        X_norm = self.feature_scaler.transform(self.X)
        self.mlkr = MLKR(n_components=2, random_state=random_state)

        start = time.time()
        print("Fitting MLKR ...")
        X_new = self.mlkr.fit_transform(X_norm, self.y)
        self.mlkr_scaler = MinMaxScaler().fit(X_new)
        raw_trans_des = self.mlkr_scaler.transform(X_new)
        end = time.time()

        print("Finished in {}".format(end-start))
        self.pca = PCA(n_components=2).fit(raw_trans_des)
        trans_des = self.pca.transform(raw_trans_des)
        self.pca_scaler = MinMaxScaler().fit(trans_des)
        if return_coords:
            return trans_des

    def transform(self, trans_df):
        X_trans = trans_df[self.pred_vars].values
        X_trans = self.feature_scaler.transform(X_trans)
        trans_gen = self.mlkr.transform(X_trans)
        trans_gen = self.mlkr_scaler.transform(trans_gen)
        trans_gen = self.pca.transform(trans_gen)
        trans_gen = self.pca_scaler.transform(trans_gen)
        return trans_gen 

class KRR():
    def __init__(self, data_df, nlfs_result, targ_var, cv_params=[3, 10]):
        self.data_df = data_df
        check_idx = np.argmin(nlfs_result["best_mae"].values)
        pred_vars = nlfs_result.iloc[check_idx]["label"].split("|")
        self.pred_vars = pred_vars
        self.X = data_df[pred_vars].values
        self.targ_var = targ_var
        self.y = data_df[targ_var].values

        params = nlfs_result.iloc[check_idx][["best_alpha", "best_gamma"]].values
        self.estimator = KernelRidge(alpha=params[0], gamma=params[1], kernel="rbf")
        self.cv_params = None
        self.cv_params = cv_params

    def fit(self):
        self.feature_scaler = MinMaxScaler().fit(self.X)
        X_norm = self.feature_scaler.transform(self.X)
        self.estimator.fit(X_norm, self.y)
    
    def predict(self, gen_data_df):
        X_pred = gen_data_df[self.pred_vars].values
        X_pred = self.feature_scaler.transform(X_pred)
        y_pred = self.estimator.predict(X_pred)
        return y_pred

if __name__=="__main__":
    gen_dir = "./input/NLFS/"
    emb_dir = gen_dir+"mlkr.pickle"
    dat_dir = "./output/exp_data/All_datasets_features.csv"
    dat_df = pd.read_csv(dat_dir, index_col=0)
#     nlfs_result = pd.read_csv(gen_dir+"All_datasets_NLFS_k.out.csv")
#     if not os.path.isfile(emb_dir):
#         embedding = Emb(dat_df, nlfs_result=nlfs_result, targ_var="k")
#         embedding.fit(return_coords=False)
#         with open(emb_dir, "wb") as handle:
#             pickle.dump(embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     else: 
#         with open(emb_dir, "rb") as handle:
#             embedding = pickle.load(handle)
#     trans_coords = embedding.transform(dat_df)
#     dat_df["x"] = trans_coords[:, 0]
#     dat_df["y"] = trans_coords[:, 1]
#     dat_df.to_csv(dat_dir)

    for tv in ["k", "G"]:
        est_dir = "./input/Estimators/krr_{}.pickle".format(tv)
        nlfs_result = pd.read_csv(gen_dir+"All_datasets_NLFS_{}.out.csv".format(tv))
        est = KRR(dat_df, nlfs_result=nlfs_result, targ_var=tv)
        print(est.pred_vars)
#         if not os.path.isfile(est_dir):
#             nlfs_result = pd.read_csv(gen_dir+"All_datasets_NLFS_{}.out.csv".format(tv))
#             est = KRR(dat_df, nlfs_result=nlfs_result, targ_var=tv)
#             est.fit()
#             with open(est_dir, "wb") as handle:
#                 pickle.dump(est, handle, protocol=pickle.HIGHEST_PROTOCOL)
#                 print("Save at {}".format(est_dir))
#         else: 
#             with open(est_dir, "rb") as handle:
#                 est = pickle.load(handle)
#         pred_prop = est.predict(dat_df)
#         print(r2_score(pred_prop, dat_df[tv].values))


