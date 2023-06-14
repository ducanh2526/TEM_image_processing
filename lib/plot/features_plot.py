import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import warnings

# import plot2 as plot 
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from sklearn.kernel_ridge import KernelRidge
from scipy.stats import pearsonr, spearmanr

time_interval = 1/30
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = {'fontname': 'serif', 'size': 20, 'labelpad': 5}
title_font = {'fontname': 'serif', 'size': 30}

size_text = 6
alpha_point = 0.7
size_point = 120

lib_dir = "./lib/"
if not lib_dir in sys.path:
    sys.path.append(lib_dir)
from general_lib import *
from VarGen import video_names
from refine_plot_mlkr import variable_signs, feature_colors, dts_colors
from estimators.kr_parameter_search import CV_predict

gen_dir = "./output/exp_data/"
dts_df = pd.read_csv(gen_dir+"/All_datasets_features.csv", index_col=0)
nlfs_dir = "./input/NLFS/"
save_dir = "./Visualize/feature_fig"
os.makedirs(save_dir)

def ax_setting():
    plt.style.use('default')
    plt.tick_params(axis='x', which='major', labelsize=15)
    plt.tick_params(axis='y', which='major', labelsize=15)
    plt.tight_layout()

def plot_regression(x, y, tv, dim, rmin=None, rmax=None, name=None, size=100, alpha=0.8, edgewidth=0.5, 
	label=None, title=None, point_color='blue', save_file=None):
	
	fig = plt.figure(figsize=(5, 5))
	ax = fig.add_subplot(111)

	plt.scatter(x, y, s=size, alpha=alpha, c=point_color, label=label,
		edgecolor="white", linewidths=edgewidth)

	y_min_plot, y_max_plot = set_plot_configuration(x=x, y=y, rmin=rmin, rmax=rmax)
	x_ref = np.linspace(y_min_plot, y_max_plot, 100)
	plt.plot(x_ref, x_ref, linestyle='-.', c='red', alpha=0.8)

	if name is not None:
		for i in range(len(name)):
			plt.annotate(str(name[i]), xy=(x[i], y[i]), size=size_text)
	if not dim=="":
		plt.ylabel(r'Observed %s (%s)' % (tv, dim), **axis_font)
		plt.xlabel(r'Predicted %s (%s)' % (tv, dim), **axis_font)
	else: 
		plt.xlabel("")
		plt.ylabel("")
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		
	plt.title(title, **title_font)
	plt.tight_layout()
	
	if save_file is not None:
		makedirs(save_file)
		plt.savefig(save_file, transparent=False)
		print ("Save at: ", save_file)
	else: 
		plt.show()
	release_mem(fig=fig)

def hist_plot(x, label='Hist 1', nbins=50, ylabel="Counts", xlabel='RBC score', 
                color="blue", score_lim = None, save_fig=None):
    fig = plt.figure(figsize=(5, 2.5))
    ax = fig.add_subplot(111)
    _, _, _ = plt.hist(x, bins=nbins, histtype='stepfilled', 
                    weights=np.zeros_like(x) + 1. / x.size,
                    density=False, label=label, log=False,  
                    color=color, edgecolor='black', 
                    alpha=0.7, linewidth=2)
    if score_lim is not None:
        plt.xticks([score_lim[0], score_lim[1]])
        scaler = MinMaxScaler().fit(np.array(score_lim).reshape(-1, 1))
        score_bound = scaler.inverse_transform(np.array([-0.05, 1.05]).reshape(-1, 1))
        score_lb = score_bound[0]
        score_ub = score_bound[1]
        plt.xlim(score_lb, score_ub)

        print(ax.get_yticklabels())
        plt.legend(loc="upper left", fontsize=20, ncol=3)
    if xlabel=="":
            ax.set_xticklabels([])
            ax.set_xlabel("")
    else:
        ax.set_xlabel(xlabel, **axis_font)
    if ylabel=="":
        ax.set_yticklabels([])
        ax.set_ylabel("")
    else:
        ax.set_ylabel(ylabel, **axis_font)
    plot.ax_setting()
    if save_fig is not None:
        if xlabel=="" and ylabel=="":
            save_fig = save_fig[:-4] + "_woAxis.pdf"
        plt.savefig(save_fig, transparent=False)
        print('Save at: {}'.format(save_fig))
    else:
        plt.show()
    release_mem(fig)

def best_est_plot(nlfs_result, data_df, targ_var, dim, tv_label=None, save_fig=None):
    # nlfs_result is a pandas.DataFrame
    start = time.time()
    # nlfs_result.sort_values(by="best_score", ascending=False, inplace=True)
    nlfs_result.sort_values(by="best_mae", ascending=True, inplace=True)
    best_pred_vars = nlfs_result.iloc[0]["label"].split("|")
    print(best_pred_vars)
    best_alpha = nlfs_result.iloc[0]["best_alpha"]
    best_gamma = nlfs_result.iloc[0]["best_gamma"]
    X = data_df[best_pred_vars].values.reshape(len(data_df), -1)
    X = MinMaxScaler().fit_transform(X)
    y_obs = data_df[targ_var].values
    krr = KernelRidge(alpha=best_alpha, gamma=best_gamma, kernel="rbf")
    y_preds =  CV_predict(krr, X, y_obs, n_folds=10, n_times=10)
    y_pred = np.mean(y_preds, axis=0)
    end = time.time()
    print("Finish CV in {}".format(end-start))
    
    if tv_label is None:
        tv_label = targ_var
    if dim=="":
        if save_fig is not None:
            save_fig = save_fig[:-4] + "_woAxis.pdf"
    plot_regression(y_pred, y_obs, tv=tv_label, dim=dim, size=100, alpha=0.8, edgewidth=0.5, save_file=save_fig)

def refine_features_plot(data_df, feature_list, feature_colors, save_fig=None):
    n_row = len(feature_list)
    n_col = 1
    plt.rcParams["axes.linewidth"] = 2

    fig = plt.figure(figsize=(8*n_col, 0.75*n_row))
    grid_width_ratio = [1 for rep in range(n_col)]
    grid = plt.GridSpec(n_row, n_col, width_ratios=grid_width_ratio)
    grid.update(wspace=0., hspace=0.1)

    frame_range = np.array([idx.replace("f", "") for idx in data_df.index]).astype(np.int32)
    frame_range = frame_range-frame_range.min()
    time_range = time_interval * frame_range
    tick_lb = 5*round(np.min(time_range)/5)
    tick_ub = 5*round(np.max(time_range)/5)
    if tick_ub>80:
        tick_step = 10
    else:
        tick_step = 5
    n_ticks = round((tick_ub-tick_lb)/tick_step)+1
    time_ticks = np.linspace(5*round(np.min(time_range)/5), 5*round(np.max(time_range)/5), n_ticks)
    
    for i, f in enumerate(feature_list):
        ax = fig.add_subplot(grid[i, 0])
        invi_features = [feature for feature in feature_list if feature!=f]
        scaler_f = MinMaxScaler().fit(data_df[f].values[:, None])
        ax.plot(time_range, minmax_scale(data_df[f].values), color=feature_colors[f], linewidth=2.5, zorder=100)
        for invi_f in invi_features:
            ax.plot(time_range, minmax_scale(data_df[invi_f].values), color="gray", alpha=0.8, linewidth=0.5)
        # ax.tick_params(length=0)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        for round_rat in np.arange(-6, 10).astype(float):
            tick_lb = int(data_df[f].values.min()*(10**round_rat))
            tick_ub = int(data_df[f].values.max()*(10**round_rat))
            if tick_lb!=tick_ub:
                tick_vals = np.array([tick_lb, tick_ub])/(10**round_rat)
                tick_pos = scaler_f.transform(tick_vals[:, None]).ravel()
                if tick_pos[0]>=-0.1 and tick_pos[1]<=1.1:
                    break 
        ax.set_yticks(tick_pos.tolist())
        ax.set_yticklabels(tick_vals.tolist())
        ax.set_ylabel(variable_signs[f], fontsize=20)
        if i<len(feature_list)-1:
            ax.set_xticks([])
        if i==len(feature_list)-1:
            ax.set_xticks(time_ticks)
            plt.tick_params(axis='x', which='major', labelsize=18, length=3)
            ax.set_xlabel("Time(s)", fontsize=20)
        plt.xlim([np.min(time_range), np.max(time_range)])
        plt.ylim([-0.2, 1.2])

    grid.tight_layout(fig, pad=0.5)
    if save_fig is not None:
        plt.savefig(save_fig)
        print('Save at: {}'.format(save_fig))
    else:
        plt.show()

def plot_corr_variables_2D(df, variable_set, n_bins=15, variable_signs=None, 
                            corr_point_ratio=15000, 
                            corr_op='pearson', save_fig=None):
    inst_idx = [int(di.split("_")[0]) for di in df.index]
    n_col = len(variable_set) 
    n_row = len(variable_set)

    fig = plt.figure(figsize=(3*n_col + 0.3, 3*n_row))
    grid_width_ratio = [1 for rep in np.arange(n_col)]
    grid = plt.GridSpec(n_row, n_col, width_ratios=grid_width_ratio)
    # fig = plt.figure(figsize=(3*(n_col+1.2), 3*n_row))
    # grid_width_ratio = [1 for rep in np.arange(n_col)]
    # grid_width_ratio.append(0.)
    # grid = plt.GridSpec(n_row, n_col+1, width_ratios=grid_width_ratio)
    grid.update(wspace=0.05, hspace=0.05)

    xx,yy = np.meshgrid(np.arange(n_col),np.arange(n_col))
    variable_pairs = np.array((xx.ravel(),yy.ravel())).T

    for i, j in variable_pairs: 
        ax = fig.add_subplot(grid[i, j]) 
        vx = variable_set[j]
        vy = variable_set[i]

        if i>j: 
            dts_color = [dts_colors[dts_idx] for dts_idx in inst_idx]
            ax_plot = ax.scatter(df[vx].values, df[vy].values, c=dts_color, s=80, alpha=0.7, marker='o', edgecolor="white")
            
        elif i<j: 
            corr_coef = round(eval('{}r'.format(corr_op))(df[vx].values, df[vy].values)[0], 3)
            marker_size = abs(corr_coef) * corr_point_ratio
            ax.scatter([.5], [.5], marker_size, [corr_coef], alpha=0.6, cmap="coolwarm",
                        vmin=-1, vmax=1, transform=ax.transAxes)
            ax.text(0.5, 0.5, '{}'.format(corr_coef), ha='center', va='center', transform=ax.transAxes, **title_font)
            plt.yticks([])
            plt.xticks([])             

        elif i==j:
            for dts_idx in set(inst_idx):
                check_index = np.where(inst_idx==dts_idx)[0]
                sns.distplot(df.iloc[check_index][vx].values, bins=n_bins, color=dts_colors[dts_idx],
                            ax=ax, hist=False,
                            kde_kws={"color": dts_colors[dts_idx], "shade":True, "lw": 1},
                            norm_hist=False)
            ax.set_ylabel('')
        
        plt.tick_params(axis='x', which='major', labelsize=12)
        plt.tick_params(axis='y', which='major', labelsize=12)
        if i==(n_row-1): 
            if variable_signs is not None:
                plt.xlabel(variable_signs[vx], **axis_font)
            else:
                plt.xlabel(vx, **axis_font)
        else:
            plt.xticks([])
        if j==0: 
            if variable_signs is not None:
                plt.ylabel(variable_signs[vy], **axis_font)
            else:
                plt.ylabel(vy, **axis_font)
        else: 
            plt.yticks([])
    # Add legend corresponding with dataset 
    # leg_ax = fig.add_subplot(grid[:, n_col]) 
    # patches = []
    # for dts_idx in set(inst_idx):
    #     label_name = "r'$\mathfrak{D}-" + "{}$'".format(dts_idx+1)
    #     patches += [mpatches.Patch(color=colors[dts_idx], label=eval(label_name), alpha=0.7)]
    # leg_ax.legend(handles=patches, loc='center left', fontsize=30)
    # plt.xticks([])
    # plt.yticks([])
    # plt.box(False)

    grid.tight_layout(fig, pad=0.5)
    if save_fig is not None: 
        plt.savefig(save_fig, bbox_inches='tight')
        print('Save at: {}'.format(save_fig))
    else:
        plt.show()

if __name__ == "__main__":
    for video_name in video_names:
        features_df = pd.read_csv("{0}/{1}/{1}_features.csv".format(gen_dir, video_name), index_col=0)
        refine_features_plot(features_df, features_list=list(variable_signs.keys()), feature_colors=feature_colors, 
                            save_fig="{0}/{1}_all_features_wLabel.pdf".format(save_dir, video_name))

    plot_corr_variables_2D(dts_df, variable_set=list(variable_signs.keys()), 
                        corr_op='spearman', corr_point_ratio=20000, variable_signs=variable_signs, 
                        save_fig=save_dir+"/feature_corr.pdf")

    tv_dict = {"k": {"name":"spring constant", "score_lim":[0.98, 1.0], "dim": r"$N/m$"}, 
               "G": {"name":"conductance", "score_lim":[0.99, 1.0], "dim": r"$G_0$"}}
    for targ_var in list(tv_dict.keys()):
        nlfs_csv = "{0}/MLKR_results/{1}/All_datasets_NLFS_{1}.out.csv".format(gen_dir, targ_var)
        nlfs_df = pd.read_csv(nlfs_csv)
        n_high_score = len(np.where(nlfs_df["best_score"].values>0.99)[0])
        print(targ_var, n_high_score, np.round(n_high_score/len(nlfs_df), 3))
        plot_config = tv_dict[targ_var]
        best_est_plot(nlfs_df, dts_df, targ_var=targ_var, 
                      dim=plot_config["dim"], 
                    #   dim="", 
                      tv_label=plot_config["name"], 
                      save_fig="{0}/best_est_{1}.pdf".format(save_dir, targ_var))
        hist_plot(nlfs_df["best_score"].values, label=targ_var, nbins=5000, score_lim=plot_config["score_lim"], 
                xlabel=r"$R^2 score$", 
                # xlabel="", ylabel="", 
                color=[feature_colors[targ_var]], 
                save_fig="{0}/All_datasets_r2_{1}.pdf".format(save_dir, targ_var))