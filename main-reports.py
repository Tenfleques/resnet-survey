#!/usr/local/bin/python3

import argparse
import os
import time
import shutil
import pandas as pd
import ast


from results import *

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Cifar100 survey')
parser.add_argument('-d', '--depth', default='20,56,110', type=str, metavar='N', help='depth of experiment whose results to investigate')
parser.add_argument('-c', '--cifar', default='10,100', type=str, metavar='N', help='cifar dataset whose results to investigate')
parser.add_argument('-p', '--params', default='lr.01,lr.02,lr.2,lr.5,lr1.0,e80,e320,mb64,mb256', type=str, help='hyper-params variants separated by commas whose results to investigate')
parser.add_argument('-m', '--metrics', default='loss_val_mean,acc_val_mean', type=str, help='metrics columns separated by commas for which the charts are based on ')
parser.add_argument('-l', '--logs', default='', type=str, help='logs default or logs-3, type for logs-3')


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
}

def main():
    args = parser.parse_args()

    list_of_hyper_param_variants = [""]

    list_of_hyper_param_variants += [params_kv.get(k) for k in args.params.split(",") if k in params_kv.keys()]

    metrics = args.metrics.split(",") 

    depths = args.depth.split(",")
    cifars = args.cifar.split(",")
    logs_dir = 'logs/'

    if args.logs:
        logs_dir = 'logs-' + args.logs + "/"


    for depth in depths:  
        for cifar in cifars:
            get_metric_logs(list_of_hyper_param_variants, metrics, depth, cifar, logs_dir)


        

def get_metric_logs(list_of_hyper_param_variants, metrics, depth, cifar, logs_dir = "logs/"):
    logs_data_train = []
    logs_data_test = []
    logs_final_prec = []    

    for hyper_param_variant in list_of_hyper_param_variants:
        name = "resnet{depth}-cifar-{cifar}{hyper_param_variant}".format(depth=depth, cifar=cifar,hyper_param_variant=hyper_param_variant)

        filename = "{logs_dir}resnet-{depth}/{name}.log".format(logs_dir=logs_dir, depth=depth, name=name)

        if not os.path.isfile(filename):
            continue
                

        dfs = get_test_and_train_logs(depth, cifar, hyper_param_variant, logs_dir)

        logs_data_test.append(dfs.get("test"))
        logs_data_train.append(dfs.get("train"))

        logs_final_prec.append({
            "name" : name,
            "cifar" : cifar,
            "layer" : depth,
            "epochs" : "",
            "batxh-size" : "",
            "lr" : "",
            "momentum" : 0.9,
            "residual-block" : "",
            "skip-connections" : "",
            "train-data-size" : "",
            "weight-decay" : "",
            "prec" : dfs.get("prec")
            # "train_stats": dfs.get("train").describe(),
            # "test_stats": dfs.get("test").describe()
        })   

    iterate_metrics(metrics, logs_data_test, depth, cifar, "test", logs_dir)
    iterate_metrics(metrics, logs_data_train, depth, cifar, "train", logs_dir)

    return

def iterate_metrics(metrics, logs_data, depth, cifar, t_set, logs_dir = "logs/"):
    len_dfs = len(logs_data)
    for metric in metrics:
        for i in range(1,len_dfs):
            # compare the base hyper-params with every other change
            logs_compare = [logs_data[0], logs_data[i]]
            print_comparison_chart(logs_compare, metric, depth, cifar, t_set, logs_dir)

        if len_dfs == 9:
            # compare the learning rates hyper params changes + the base 
            print_comparison_chart(logs_data[:6], metric, depth, cifar, t_set, logs_dir)

            # compare the epochs hyper params changes + the base 
            print_comparison_chart([logs_data[0]] +  logs_data[6:8], metric, depth, cifar, t_set, logs_dir)

            # compare the batch-size hyper params changes + the base 
            print_comparison_chart([logs_data[0]] + logs_data[8:], metric, depth, cifar, t_set, logs_dir)

def print_comparison_chart(logs_compare, metric, depth, cifar, t_set, logs_dir = "logs/"):
    chart_path = "{logs_dir}charts/resnet-{depth}".format(logs_dir=logs_dir, depth=depth)

    filename = "cifar-{cifar}-{t}{f}".format(t=t_set, f="_".join([a.get("name") for a in logs_compare] + [metric]), cifar=cifar)

    os.makedirs(chart_path, exist_ok=True)


    comparison_chart = compare_dfs(logs_compare, [metric], chart_path + "/" + filename)

    return comparison_chart


if __name__=='__main__':
    main()

