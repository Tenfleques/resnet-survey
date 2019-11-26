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
parser.add_argument('-p', '--params', default='lr.01,lr.02,lr.2,lr.5,lr1.0,e80,e320,mb64,mb256,rb0,rb2,rb3,sc1,sc3,td5,td2', type=str, help='hyper-params variants separated by commas whose results to investigate')
parser.add_argument('-m', '--metrics', default='loss_val_mean,acc_val_mean,acc', type=str, help='metrics columns separated by commas for which the charts are based on ')
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

    depths = args.depth.split(",")
    cifars = args.cifar.split(",")
    logs_dir = 'logs/'

    if args.logs:
        logs_dir = 'logs-' + args.logs + "/"
    
    run(depths=depths, 
            cifars=cifars, 
            hyper_params=args.params,
            metrics=args.metrics,
            logs_dir='logs/'
            )

def run(depths=[20,56,110], 
            cifars=[10,100], 
            hyper_params='lr.01,lr.02,lr.2,lr.5,lr1.0,e80,e320,mb64,mb256',
            metrics='loss_val_mean,acc_val_mean',
            logs_dir='logs/'
            ):

    metrics = metrics.split(",")

    list_of_hyper_param_variants = [""]

    list_of_hyper_param_variants += [params_kv.get(k) for k in hyper_params.split(",") if k in params_kv.keys()]

    logs_table = []
    for depth in depths:  
        for cifar in cifars:
            logs_table += get_metric_logs(list_of_hyper_param_variants, metrics, depth, cifar, logs_dir)
    
    return logs_table
       

def get_metric_logs(list_of_hyper_param_variants, metrics, depth, cifar, logs_dir = "logs/"):
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

        hyper_keys_dict = {
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
            hyper_key_arr = hyper_param_variant.split("-")
            hyper_key_val = hyper_key_arr.pop()
            hyper_key_key = ''.join(hyper_key_arr[1:])
            hyper_keys_dict[hyper_key_key] = hyper_key_val
            

        logs_table.append(hyper_keys_dict)   
 

    iterate_metrics(metrics, logs_data_test, depth, cifar, "test", logs_dir)
    iterate_metrics(metrics, logs_data_train, depth, cifar, "train", logs_dir)

    return logs_table

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

    if not os.path.isfile(chart_path + "/" + filename + ".png"):
        comparison_chart = compare_dfs(logs_compare, [metric], chart_path + "/" + filename)
        return comparison_chart
    return


if __name__=='__main__':
    main()

