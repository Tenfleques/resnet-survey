#!/usr/local/bin/python3
import argparse
import os
import time
import shutil

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Cifar100 survey')
parser.add_argument('-d', '--depth', default=20, type=int, metavar='N', help='depth of experiment whose results to investigate')
parser.add_argument('-c', '--cifar', default=10, type=int, metavar='N', help='cifar dataset whose results to investigate')

def main():
    global args
    args = parser.parse_args()

def read_model(depth, cifar):
    filename = "".format(depth=depth, cifar=cifar)
    file_handle = open()

if __name__=='__main__':
    main()