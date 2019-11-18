#!/usr/local/bin/python3
import argparse
import os
import time
import shutil

import ast
 

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Cifar100 survey')
parser.add_argument('-d', '--depth', default=20, type=int, metavar='N', help='depth of experiment whose results to investigate')
parser.add_argument('-c', '--cifar', default=10, type=int, metavar='N', help='cifar dataset whose results to investigate')
parser.add_argument('-t', '--task', default='', type=str, help='task parameter whose results to investigate')

def main():
    args = parser.parse_args()
    read_logs(args.depth, args.cifar, args.task)

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

def read_logs(depth, cifar, task):
    filename = "./resnet-{depth}/resnet{depth}-cifar-{cifar}{task}.log".format(depth=depth, cifar=cifar, task=task)

    if not os.path.isfile(filename):
        return

    file_handle = open(filename, "r")
    file_contents = file_handle.readlines()

    epochs = []
    tests = []

    train_batch = []
    test_batch = []

    for line in file_contents:
        if 'Epoch' in line:
            train_batch.append(parse_epoch(line))
            if len(test_batch):
                tests.append(test_batch)

            test_batch = []
        if 'Test' in line:
            test_batch.append(parse_test(line))
            if len(train_batch):
                epochs.append(train_batch)

            train_batch = []

    print(epochs[2])
    print(tests[2])

if __name__=='__main__':
    main()