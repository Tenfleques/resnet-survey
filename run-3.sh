#!/bin/bash

OPTIND=1

depth=20
ct=10
residual_block=-1
skip_connections=0
def_res=-1
def_sc=0
def_tds=10
train_data_size=10



while getopts "h?d:c:r:s:t:" opt; do
    case "$opt" in
    h|\?)
        echo "-d <depth of the network (20|50|56|62|110), default = 20> -c <cifar dataset to use (10|100), default 10> -s <skip connections (-1|1|3), default -1> -r <residual blocks (-1|0|2|3), default -1> -t <training data size(10|5|2), default 10>"
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
    r)  if [[ "$OPTARG" =~ ^(-1|0|2|3)$ ]]; then
            residual_block=$OPTARG
        fi
        ;;
    s)  if [[ "$OPTARG" =~ ^(-1|1|3)$ ]]; then
            skip_connections=$OPTARG
        fi
        ;;
    t)  if [[ "$OPTARG" =~ ^(10|5|2)$ ]]; then
            train_data_size=$OPTARG
        fi
        ;;
    esac
done


# run resnet-${depth}-cifar{ct} with default params
if [ ! -f public/logs/resnet-${depth}/resnet${depth}-cifar-${ct}.log ]; then
    echo "resnet-${depth} dataset-${ct} epochs 160, batch-size 128, learning rate 0.1, ground paramaters "

    CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > public/logs/resnet-${depth}/resnet${depth}-cifar-${ct}.log

fi

if [ "$residual_block" -ne "$def_res"  ]; then
    echo " Adjusting the residual-blocks ${residual_block}"

    if [ ! -f public/logs/resnet-${depth}/resnet${depth}-cifar-${ct}-rb-${residual_block}.log ]; then
    # adjust the residual-blocks
        CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} -rb ${residual_block} > public/logs/resnet-${depth}/resnet${depth}-cifar-${ct}-rb-${residual_block}.log

    fi
fi

if [ "$skip_connections" -ne "$def_sc"  ]; then
    echo " Adjusting the skipping connections = ${skip_connections}"

    if [ ! -f public/logs/resnet-${depth}/resnet${depth}-cifar-${ct}-sc-${skip_connections}.log ]; then
    # adjust the skip-connections
        CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} -sc ${skip_connections} > public/logs/resnet-${depth}/resnet${depth}-cifar-${ct}-sc-${skip_connections}.log

    fi
fi

if [ "$train_data_size" -ne "$def_tds"  ]; then

    train_data_size=$(awk -v ts=$train_data_size 'BEGIN { print (ts / 10) }')

    echo " Adjusting the training data size connections = ${train_data_size} "

    if [ ! -f public/logs/resnet-${depth}/resnet${depth}-cifar-${ct}-tds-${train_data_size}.log ]; then
    #  adjust the skip-connections
        CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} -tds ${train_data_size} > public/logs/resnet-${depth}/resnet${depth}-cifar-${ct}-tds-${train_data_size}.log

    fi
fi

echo " finished "