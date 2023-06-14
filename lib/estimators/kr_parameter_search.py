import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge

#############################################################
#   Generate n_times trials on learning model from data
#   Data is splited into trainning set and test set
#   Training set is used for trainning model
#   Test set is used for prediction
#   Return
def CV_predict(model, X, y, n_folds=3, n_times=3, rand_idx=0):
    if (n_folds <= 0) or (n_folds > len(y)):
        n_folds = len(y)
        n_times = 1

    y_predicts = []
    for i in range(n_times):
        np.random.seed(rand_idx+i)
        indexes = np.random.permutation(range(len(y)))
        kf = KFold(n_splits=n_folds)

        y_cv_predict = []
        cv_test_indexes = []
        cv_train_indexes = []
        for train, test in kf.split(indexes):
            cv_test_indexes += list(indexes[test])

            X_train, X_test = X[indexes[train]], X[indexes[test]]
            y_train, Y_test = y[indexes[train]], y[indexes[test]]

            model.fit(X_train, y_train)

            y_test_predict = model.predict(X_test)
            y_cv_predict += list(y_test_predict)

        cv_test_indexes = np.array(cv_test_indexes)
        rev_indexes = np.argsort(cv_test_indexes)

        y_cv_predict = np.array(y_cv_predict)

        y_predicts += [y_cv_predict[rev_indexes]]

    y_predicts = np.array(y_predicts)

    return y_predicts

def CV_predict_score(model, X, y, n_folds=3, n_times=3):

    if (n_folds <= 0) or (n_folds > len(y)):
        n_folds = len(y)
        n_times = 1

    y_predicts = []
    scores = []
    errors = []
    for i in range(n_times):
        y_predict = CV_predict(model=model, X=X, y=y, n_folds=n_folds, n_times=1, rand_idx=i)
        # n_times = 1 then the result has only 1 y_pred array
        y_predict = y_predict[0]
        this_r2 = r2_score(y_true=y, y_pred=y_predict)
        this_err = mean_absolute_error(y_true=y, y_pred=y_predict)
        scores.append(this_r2)
        errors.append(this_err)

    return np.mean(scores), np.std(scores), np.mean(errors), np.std(errors)


def kernel_ridge_parameter_search(X, y_obs, kernel='rbf',
                                  n_folds=3, n_times=3):
    # parameter initialize
    gamma_log_lb = -2.0 
    gamma_log_ub = 2.0 
    alpha_log_lb = -4.0 
    alpha_log_ub = -1.0 
    n_steps = 20
    n_rounds = 4
    alpha = 1
    gamma = 1
    lb = 0.8
    ub = 1.2
    n_instance = len(y_obs)

    if (n_folds <= 0) or (n_folds > n_instance):
        n_folds = n_instance
        n_times = 1

    # Start
    for i in range(n_rounds):
        scores_mean = []
        scores_std = []
        gammas = np.logspace(gamma_log_lb, gamma_log_ub, num=n_steps)
        for gamma in gammas:
            k_ridge = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel)
            y_predicts = CV_predict(
                k_ridge, X, y_obs, n_folds=n_folds, n_times=n_times)
            cv_scores = list(map(lambda y_predict: r2_score(
                y_obs, y_predict), y_predicts))

            scores_mean += [np.mean(cv_scores)]
            scores_std += [np.std(cv_scores)]

        best_index = np.argmax(scores_mean)
        gamma = gammas[best_index]
        gamma_log_lb = np.log10(gamma * lb)
        gamma_log_ub = np.log10(gamma * ub)
        scores_mean = []
        scores_std = []
        alphas = np.logspace(alpha_log_lb, alpha_log_ub, num=n_steps)
        for alpha in alphas:
            k_ridge = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel)
            y_predicts = CV_predict(
                k_ridge, X, y_obs, n_folds=n_folds, n_times=n_times)
            cv_scores = list(map(lambda y_predict: r2_score(
                y_obs, y_predict), y_predicts))

            scores_mean += [np.mean(cv_scores)]
            scores_std += [np.std(cv_scores)]

        best_index = np.argmax(scores_mean)
        alpha = alphas[best_index]
        alpha_log_lb = np.log10(alpha * lb)
        alpha_log_ub = np.log10(alpha * ub)

    return alpha, gamma, scores_mean[best_index], scores_std[best_index]

