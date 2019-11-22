#!/usr/local/bin/python

import argparse
import os
import time
import shutil
import pandas as pd
import ast


from results import *

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Cifar100 survey')
parser.add_argument('-d', '--depth', default=20, type=int, metavar='N', help='depth of experiment whose results to investigate')
parser.add_argument('-c', '--cifar', default=10, type=int, metavar='N', help='cifar dataset whose results to investigate')
parser.add_argument('-p', '--params', default='-lr-.01,-lr-.02,-lr-.2,-lr-.5,-lr-1.0,-epochs-80,-epochs-320,-mini-batch-64,-mini-batch-256', type=str, help='hyper-params variants separated by commas whose results to investigate')
parser.add_argument('-m', '--metrics', default='loss_val_mean,acc_val_mean', type=str, help='metrics columns separated by commas for which the charts are based on ')


def main():
    args = parser.parse_args()

    list_of_hyper_param_variants = [""]

    list_of_hyper_param_variants += args.params.split(",") 

    metrics = args.metrics.split(",") 

    logs_data_train = []
    logs_data_test = []
    depth = 20
    cifar = 10

    for hyper_param_variant in list_of_hyper_param_variants:
        filename = "./resnet-{depth}/resnet{depth}-cifar-{cifar}{hyper_param_variant}.log".format(depth=depth, cifar=cifar, hyper_param_variant=hyper_param_variant)

        if not os.path.isfile(filename):
            continue
        
        print(hyper_param_variant)
        dfs = get_test_and_train_logs(depth, cifar, hyper_param_variant)

        logs_data_test.append(dfs.get("test"))
        logs_data_train.append(dfs.get("train"))


    len_dfs = len(logs_data_test)

    for metric in metrics:
        for i in range(1,len_dfs):
            # compare the base hyper-params with every other change
            logs_compare = [logs_data_test[0], logs_data_test[i]]
            print_comparison_chart(logs_compare, metric, depth)
        # compare the learning rates hyper params changes + the base 
        print_comparison_chart(logs_data_test[:6], metric, depth)

        if len_dfs == 9:
            # compare the epochs hyper params changes + the base 
            print_comparison_chart([logs_data_test[0]] +  logs_data_test[6:8], metric, depth)

            # compare the batch-size hyper params changes + the base 
            print_comparison_chart([logs_data_test[0]] + logs_data_test[8:], metric, depth)

def print_comparison_chart(logs_compare,metric, depth):
    chart_path = "./resnet-{depth}/charts/{f}".format(depth=depth, f="_".join([a.get("name") for a in logs_compare] + [metric]))

    comparison_chart = compare_dfs(logs_compare, [metric], chart_path)

    return comparison_chart


if __name__=='__main__':
    main()

