#!/usr/local/bin/python3

import argparse
import os
import time
import shutil
import pandas as pd
import ast
import matplotlib.pyplot as plt


def get_test_and_train_logs(depth, cifar, hyper_param, logs_dir = "logs/"):
    data = read_logs(depth, cifar, hyper_param, logs_dir)

    test_summarised_df = prepare_summary_df(data.get("test"))
    train_summarised_df = prepare_summary_df(data.get("train"))

    df_test = prepare_df_object(test_summarised_df.get("df"), hyper_param )
    df_train = prepare_df_object(train_summarised_df.get("df"), hyper_param ) 
    
    return {
        "train" : df_train,
        "test"  : df_test,
        "prec"  : data.get("final_accuracy"),
        "train_mean_avg_loss" : train_summarised_df.get("mean_avg_loss"),
        "train_mean_val_loss" : train_summarised_df.get("mean_val_loss"),
        "test_mean_avg_loss" : test_summarised_df.get("mean_avg_loss"),
        "test_mean_val_loss" : test_summarised_df.get("mean_val_loss"),
    }

def read_logs(depth=20, cifar=10, hyper_param="", logs_dir = "logs/"):
    filename = "{logs_dir}resnet-{depth}/resnet{depth}-cifar-{cifar}{hyper_param}.log".format(logs_dir=logs_dir,depth=depth, cifar=cifar, hyper_param=hyper_param)

    if not os.path.isfile(filename):
        return

    file_handle = open(filename, "r")
    file_contents = file_handle.readlines()

    train_batches = []
    test_batches = []

    train_batch = []
    test_batch = []
    
    acc = 0
    for line in file_contents:
        if 'Epoch' in line:
            train_batch.append(parse_epoch(line))
            if len(test_batch):
                test_batches.append(test_batch)

            test_batch = []
        if 'Test' in line:
            test_batch.append(parse_test(line))
            if len(train_batch):
                train_batches.append(train_batch)

            train_batch = []
        if "Prec" in line:
            acc = line.strip().replace("%","").split()[2]

    return {
        "train" : train_batches,
        "test"   : test_batches,
        "final_accuracy": acc
    }

def parse_epoch(line):
    """
        process the epoch line, get epoch number, iteration number, time, data, loss, precision 
    """
    # Epoch: [1][300/391]	Time 0.033 (0.019)	Data 0.020 (0.004)	Loss 1.0580 (1.1237)	Prec 64.062% (59.437%)
    pieces = line.split("\t")

    epoch_id = ast.literal_eval("(" + pieces[0].split()[1].replace("][", "],[").replace("/", "],[") + ")")

    batch_time_val, batch_time_avg = ast.literal_eval("(" + pieces[1].replace(" (", ",").split()[1])
    data_time_val, data_time_avg  = ast.literal_eval("(" + pieces[2].replace(" (", ",").split()[1])
    loss_val, loss_avg = ast.literal_eval("(" + pieces[3].replace(" (", ",").split()[1])
    acc_val, acc_avg = ast.literal_eval("(" + pieces[4].replace(" (", ",").replace("%","").split()[1])

    
    epoch = {
        "epoch_id" : epoch_id[0][0],
        "iteration" : epoch_id[1][0],
        "len_trainloader" : epoch_id[2][0],
        "time_val" : batch_time_val,
        "time_avg" : batch_time_avg,
        "data_time_val" : data_time_val,
        "data_time_avg" : data_time_avg,
        "loss_val" : loss_val,
        "loss_avg" : loss_avg,
        "acc_val" : acc_val,
        "acc_avg" : acc_avg,
    }
    return epoch


def parse_test(line):
    """
        process the test line, iteration number, time, loss, precision 
    """
    # template = 
    # Test: [0/100]	Time 0.113 (0.113)	Loss 0.8992 (0.8992)	Prec 65.000% (65.000%)    
    pieces = line.split("\t")

    test_id = ast.literal_eval(pieces[0].split()[1].replace("/", ","))
    batch_time_val, batch_time_avg = ast.literal_eval("(" + pieces[1].replace(" (", ",").split()[1])
    loss_val, loss_avg = ast.literal_eval("(" + pieces[2].replace(" (", ",").split()[1])
    acc_val, acc_avg = ast.literal_eval("(" + pieces[3].replace(" (", ",").replace("%","").split()[1])
    
    test = {
        "test_id" : test_id,
        "time_val" : batch_time_val,
        "time_avg" : batch_time_avg,
        "loss_val" : loss_val,
        "loss_avg" : loss_avg,
        "acc_val" : acc_val,
        "acc_avg" : acc_avg,
    }

    return test

def prepare_summary_df(dfs):
    stats_col_params = ["mean","std", "min", "max"]
    stats_df_src = []
    for batch in dfs:
        stats = pd.DataFrame(batch).describe().loc[stats_col_params]
        new_series = {}
        for i in stats.columns:
            for j in stats_col_params:
                new_series[i+"_"+j] = stats.loc[j][i]

        stats_df_src.append(new_series)

    df = pd.DataFrame(stats_df_src)
    desc_df = df.describe()
    return {    
        "df" : df,
        "mean_avg_loss" : desc_df["loss_avg_mean"]["mean"],
        "mean_val_loss" : desc_df["loss_val_mean"]["mean"],
    }


# prepares the dict of dataframe object 

def prepare_df_object(df, name ):
    return {
        "name" : name,
        "stats" : df.describe(),
        "df" : df,
        "mean_val_loss" : df.describe().loc["mean"]["loss_val_mean"],
        "mean_avg_loss" : df.describe().loc["mean"]["loss_avg_mean"]
    }


def compare_dfs(list_of_dicts_of_dfs, list_params, print_to_file = ""):
    """ compare  reults of different logs """

    # find the dfs intersection
    # find the mean sqaure difference
    # find the min and max difference 

    param_comparison = {}

    # plot the comparisons

    for param in list_params:
        plot_diff = pd.DataFrame()
        for df_obj in list_of_dicts_of_dfs:
            hyper = df_obj.get("name")
            if not hyper:
                hyper = "base"
            plot_diff[ hyper + "_" + param] = df_obj.get("df")[param]
       
        
        if print_to_file and plot_diff.shape[0]:
            plot = plot_diff.plot()

            plt.title(param )
            plt.savefig(print_to_file + ".png", dpi=144)
            plt.close() 
        
        param_comparison[param] = plot_diff


            # plot_log = plot_diff.plot(logy=True) 

            # plt.title('log-graph ' + param)
            # plt.savefig(print_to_file + "-log.png")
            # plt.close() 


    return param_comparison
