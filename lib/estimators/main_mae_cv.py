import sys
import os
import numpy as np
import pandas as pd
import warnings
from argparse import ArgumentParser
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_ridge import KernelRidge
from kr_parameter_search import CV_predict_score, kernel_ridge_parameter_search

video_names = ['11278793', '1423452', '12107372', '12101805', '12094593']

warnings.filterwarnings("ignore")

def local_krr_cv(data, target_variable, predicting_variables, best_alpha=None, best_gamma=None,
                kernel='rbf', n_folds=10, n_times=10):
    print(predicting_variables, target_variable)
    if target_variable in predicting_variables:
        predicting_variables.remove(target_variable)

    X = data[predicting_variables].values.reshape(len(data), -1)
    y_obs = data[target_variable].values.reshape(len(data), )

    min_max_scaler_X = MinMaxScaler()
    X = min_max_scaler_X.fit_transform(X)
    if (best_alpha is None) and (best_gamma is None):
        best_alpha, best_gamma, best_score, best_score_std = \
            kernel_ridge_parameter_search(X, y_obs, kernel=kernel,
                                        n_folds=n_folds, n_times=n_times)
    krr = KernelRidge(alpha=best_alpha, gamma=best_gamma, kernel=kernel)
    r2_score, r2_score_std, mae, mae_std =  CV_predict_score(krr, X, y_obs, n_folds=n_folds, n_times=n_times)
    result_dict = result_dict = {"best_alpha":best_alpha, "best_gamma":best_gamma, "best_score":r2_score, "best_score_std":r2_score_std,
                                "best_mae":mae, "best_mae_std":mae_std, "label":"|".join(pv for pv in predicting_variables)}
    return result_dict

def parse_arguments():
    parser = ArgumentParser(description="CV-MAE")
    parser.add_argument("--input_dat", type=str, required=True,
                        help="Directory to the file containing dataset")
    parser.add_argument("--nlfs_dat", type=str, required=True,
                        help="Directory to the NLFS result file containing hyperparameters")
    parser.add_argument("--targ_var", type=str, nargs="?", default="k", 
                        help="Target variable used for estimation")
    parser.add_argument("--n_cv", type=int, nargs="?", default=10,
                        help="Number of folds used for CV")
    parser.add_argument("--n_times", type=int, nargs="?", default=10,
                        help="Number of times running CV")
    return parser.parse_args()

def main():
    from pyspark import SparkContext

    args = parse_arguments()

    start_time = datetime.now()

    target_variable = args.targ_var
    data_df = pd.read_csv(args.input_dat, index_col=0)
    out_dir = os.path.join(*args.nlfs_dat.split("/")[:-1])

    # nlfs_csv = "{0}/All_datasets/All_datasets_features_{1}_out.csv".format(config["input_dir"], target_variable)
    nlfs_result_df = pd.read_csv(args.nlfs_dat, index_col="label")
    krr_params = []
    for pv in nlfs_result_df.index[:]:
        krr_params += [{"pv": pv.split("|"), "alpha": nlfs_result_df.loc[pv, "best_alpha"], "gamma": nlfs_result_df.loc[pv, "best_gamma"]}]

    elements = list(enumerate(krr_params))
    # TODO: backup things like this
    """
    if os.path.exists(config["output"]["file"]):
        df = pd.read_csv(co nfig["input"]["file"])
        csv_nfc = df.shape[0]
    #     # with open("{}.crash".format(uuid.uuid4().hex), "w") as f:
    #     #     f.write("{}".format(csv_nfc))
        fcg = fcg[csv_nfc:]

    """ 

    duration = (datetime.now() - start_time).total_seconds()
    print("Time for pre-processing data: {} (s)".format(duration))

    ##
    # Spark
    ##
    start_time = datetime.now()

    sc = SparkContext(appName="CV_MAE")
    n_eles = len(elements)
    print("Number of Initial Vectors: {0}".format(n_eles))

    try:
        # The data is big, thus broadcast it to node.
        data = sc.broadcast(data_df)
        groups = [elements[i:i + 2**13]
                  for i in range(0, len(elements), 2**13)]

        for group in groups:
            # Divide feature combinations to the number of partitions.
            n_eles = len(group)
            if n_eles < args.partitions:
                rdd = sc.parallelize(group, numSlices=n_eles)
            else:
                rdd = sc.parallelize(group, numSlices=args.partitions)

            rs_regression = rdd.map(
                lambda e: (e[1], local_krr_cv(data=data.value, target_variable=target_variable, predicting_variables=e[1]["pv"], 
                                            best_alpha=e[1]["alpha"], best_gamma=e[1]["gamma"],
                                            kernel='rbf', n_folds=args.n_cv, n_times=args.n_times)))
            result = rs_regression.collect()

            ##
            # Write the result
            ##
            results = [r[1] for r in result]
            output = pd.DataFrame.from_records(map(lambda x: x, results))
            output_file = "{0}/All_datasets_NLFS_{1}.out.csv".format(out_dir, args.targ_var)
                                                                    
            if not os.path.exists(output_file):
                output.to_csv(output_file, index=False)
                print("Output file: {}".format(output_file))
            else:
                output.to_csv(output_file, index=False, mode="a", header=False)
                print("Appended to {}".format(output_file))
    except KeyboardInterrupt:
        print("\nProgram terminated by Ctrl +C ")
    finally:
        sc.stop()

    duration=(datetime.now() - start_time).total_seconds()
    print("Time for running on Spark: {} (s)".format(duration))


if __name__ == "__main__":
    sys.exit(main())
    # data = pd.read_csv("./output/exp_data/All_datasets_features.csv", index_col=0)
    # nlfs_df = pd.read_csv("./input/NLFS/All_datasets_NLFS_k.out.csv", index_col="label")
    # check_idx = np.argmin(nlfs_df["best_mae"].values)
    # pred_vars = nlfs_df.index[check_idx].split("|")
    # best_alpha, best_gamma = nlfs_df.iloc[check_idx][["best_alpha", "best_gamma"]].values
    # print(nlfs_df.iloc[check_idx])

    # result_dict = local_krr_cv(data=data, target_variable="k", 
    #                         predicting_variables=pred_vars, 
    #                         best_alpha=best_alpha, best_gamma=best_gamma,
    #                         kernel='rbf', n_folds=10, n_times=10)
    # print(result_dict)