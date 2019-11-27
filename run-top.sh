#!/bin/bash

OPTIND=1

depth=20
ct=10
ep=320
mb=64

while getopts "h?d:c:e:m:" opt; do
    case "$opt" in
    h|\?)
        echo "-d <depth of the network. default = 20> -c <cifar dataset to use, default 10>"
        exit 0
        ;;
    d)  if [[ "$OPTARG" =~ ^(20|50|56|62|110)$ ]]; then
            depth=$OPTARG
        fi
        ;;
    c)  if [[ "$OPTARG" =~ ^(10|100)$ ]]; then
            ct=$OPTARG
        fi
        ;;
    e)  if [[ "$OPTARG" =~ ^(80|160|320)$ ]]; then
            ep=$OPTARG
        fi
        ;;
    m)  if [[ "$OPTARG" =~ ^(64|128|256)$ ]]; then
            mb=$OPTARG
        fi
        ;;
    esac
done


# run top performing hyper-params for precision and loss
if [ ! -f public/logs/resnet-${depth}/resnet${depth}-cifar-${ct}-epochs-${ep}-mini-batch-${mb}.log ]; then
    echo "########## resnet-${depth} dataset-${ct} epochs ${ep} , batch-size ${mb} , learning rate 0.1 #########"

    CUDA_VISIBLE_DEVICES=0 python main.py --epoch ${ep} --batch-size ${mb} --lr 0.1 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > public/logs/resnet-${depth}/resnet${depth}-cifar-${ct}-epochs-${ep}-mini-batch-${mb}.log

fi


