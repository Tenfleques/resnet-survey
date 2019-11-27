#!/usr/local/bin/python3

import argparse
import os
import time
import shutil
import pandas as pd
import ast

from results import *

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Cifar100 survey')
parser.add_argument('-d', '--depth', default='20,50,56,62,110', type=str, metavar='N', help='depth of experiment whose results to investigate')
parser.add_argument('-c', '--cifar', default='10,100', type=str, metavar='N', help='cifar dataset whose results to investigate')
parser.add_argument('-p', '--params', default='l50,l62,lr.01,lr.02,lr.2,lr.5,lr1.0,e80,e320,mb64,mb256,rb0,rb2,rb3,sc1,sc3,td5,td2', type=str, help='hyper-params variants separated by commas whose results to investigate')
parser.add_argument('-m', '--metrics', default='loss_val_mean,acc_val_mean', type=str, help='metrics columns separated by commas for which the charts are based on ')
parser.add_argument('-l', '--logs', default='logs/', type=str, help='logs default or logs-3, type for logs-3')
parser.add_argument('-g', '--graphs', default=1, type=int, help='whether to plot graphs or not')


params_kv = {
    "lr.01"    : "-lr-.01",
    "lr.02"    : "-lr-.02",
    "lr.2"     : "-lr-.2",
    "lr.5"     : "-lr-.5",
    "lr1.0"    : "-lr-1.0",
    "e80"      : "-epochs-80", 
    "e320"     : "-epochs-320",
    "mb64"     : "-mini-batch-64",
    "mb256"    : "-mini-batch-256",
    "rb0"      : "-rb-0",
    "rb2"      : "-rb-2",
    "rb3"      : "-rb-3",
    "sc1"      : "-sc-1",
    "sc3"      : "-sc-3",
    "td5"      : "-tds-0.5",
    "td2"      : "-tds-0.2",
    "l50"      : "-l50",
    "l62"      : "-l62"
}

def main():
    args = parser.parse_args() 

    depths = args.depth.split(",")
    cifars = args.cifar.split(",")

    
    
    run(depths=depths, 
            cifars=cifars, 
            hyper_params=args.params,
            metrics=args.metrics,
            logs_dir=args.logs,
            charts = args.graphs
            )

def run(depths=[20,50,56,62,110], 
            cifars=[10,100], 
            hyper_params='lr.01,lr.02,lr.2,lr.5,lr1.0,e80,e320,mb64,mb256,rb0,rb2,rb3,sc1,sc3,td5,td2,l50,l62',
            metrics='loss_val_mean,acc_val_mean',
            logs_dir='logs/',
            charts=False
            ):

    metrics = metrics.split(",")

    list_of_hyper_param_variants = [""]

    list_of_hyper_param_variants += [params_kv.get(k) for k in hyper_params.split(",") if k in params_kv.keys()]

    logs_table = []
    for depth in depths:  
        for cifar in cifars:
            logs_table += get_metric_logs(list_of_hyper_param_variants, metrics, depth, cifar, logs_dir, charts)
    
    return logs_table
       

def get_metric_logs(list_of_hyper_param_variants, metrics, depth, cifar, logs_dir = "logs/", charts = False):
    logs_data_train = []
    logs_data_test = []
    logs_table = []    

    for hyper_param_variant in list_of_hyper_param_variants:
        name = "resnet{depth}-cifar-{cifar}{hyper_param_variant}".format(depth=depth, cifar=cifar,hyper_param_variant=hyper_param_variant)

        filename = "{logs_dir}resnet-{depth}/{name}.log".format(logs_dir=logs_dir, depth=depth, name=name)

        if not os.path.isfile(filename):
            continue
                

        dfs = get_test_and_train_logs(depth, cifar, hyper_param_variant, logs_dir)

        logs_data_test.append(dfs.get("test"))
        logs_data_train.append(dfs.get("train"))

        helper_kvs_dict = {
            "name" : name,
            "cifar" : cifar,
            "layer" : depth,
            "epochs" : 160,
            "minibatch" : 128,
            "lr" : 0.1,
            "rb" : -1,
            "sc" : -1,
            "tds" : 1.0,
            "prec" : dfs.get("prec"),
            "train_avg_loss" : dfs.get("train_mean_avg_loss"),
            "train_val_loss" : dfs.get("train_mean_val_loss"),
            "test_avg_loss" : dfs.get("test_mean_avg_loss"),
            "test_val_loss" : dfs.get("test_mean_avg_loss")
        }

        if hyper_param_variant in params_kv.values():
            helper_kv_arr = hyper_param_variant.split("-")
            helper_kv_val = helper_kv_arr.pop()
            helper_kv_key = ''.join(helper_kv_arr[1:])
            helper_kvs_dict[helper_kv_key] = helper_kv_val
            

        logs_table.append(helper_kvs_dict)   
 
    if charts:
        iterate_metrics(metrics, logs_data_test, depth, cifar, "test", dfs.get("prec"), logs_dir)
        iterate_metrics(metrics, logs_data_train, depth, cifar, "train", dfs.get("prec"), logs_dir)

    return logs_table

def iterate_metrics(metrics, logs_data, depth, cifar, t_set, prec, logs_dir = "logs/"):
    len_dfs = len(logs_data)
    for metric in metrics:
        for i in range(1,len_dfs):
            # compare the base hyper-params with every other change
            logs_compare = [logs_data[0], logs_data[i]]
            print_comparison_chart(logs_compare, metric, depth, cifar, t_set, prec)

        if len_dfs == 9:
            # compare the learning rates hyper params changes + the base 
            print_comparison_chart(logs_data[:6], metric, depth, cifar, t_set, prec)

            # compare the epochs hyper params changes + the base 
            print_comparison_chart([logs_data[0]] +  logs_data[6:8], metric, depth, cifar, t_set, prec)

            # compare the batch-size hyper params changes + the base 
            print_comparison_chart([logs_data[0]] + logs_data[8:], metric, depth, cifar, t_set, prec)

def print_comparison_chart(logs_compare, metric, depth, cifar, t_set, prec, logs_dir = "public/"):
    chart_path = "{logs_dir}charts/resnet-{depth}".format(logs_dir=logs_dir, depth=depth)

    filename = "cifar-{cifar}-{t}{f}".format(t=t_set, f="_".join([a.get("name") for a in logs_compare] + [metric]), cifar=cifar)

    os.makedirs(chart_path, exist_ok=True)

    if not os.path.isfile(chart_path + "/" + filename + ".png"):
        comparison_chart = compare_dfs(logs_compare, [metric], prec, chart_path + "/" + filename)
        return comparison_chart
    return


if __name__=='__main__':
    main()

