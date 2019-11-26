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

while getopts "h?d:c:" opt; do
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


# run resnet-${depth} with default params
if [ ! -f logs/resnet-${depth}/resnet${depth}-cifar-${ct}.log ]; then
    echo "########## resnet-${depth} dataset-${ct} epochs 160, batch-size 128, learning rate 0.1, ground paramaters #########"

    CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}.log

fi

echo "########## Adjusting the learning rate #############"

if [ ! -f logs/resnet-${depth}/resnet${depth}-cifar-${ct}-lr-1.0.log ]; then
# adjust the learning rate *=10
    CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr 1.0 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-lr-1.0.log

fi


if [ ! -f logs/resnet-${depth}/resnet${depth}-cifar-${ct}-lr-.5.log ]; then
    # adjust the learning rate *=5
    CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr .5 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-lr-.5.log

fi

if [ ! -f logs/resnet-${depth}/resnet${depth}-cifar-${ct}-lr-.2.log ]; then
    # adjust the learning rate *=2
    CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr .2 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-lr-.2.log

fi

if [ ! -f logs/resnet-${depth}/resnet${depth}-cifar-${ct}-lr-.01.log ]; then
    # adjust the learning rate *=.1
    CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr .01 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-lr-.01.log
fi 

if [ ! -f logs/resnet-${depth}/resnet${depth}-cifar-${ct}-lr-.02.log ]; then
    # adjust the learning rate *=.2
    CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr .02 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-lr-.02.log
fi 

if [ ! -f logs/resnet-${depth}/resnet${depth}-cifar-${ct}-lr-.05.log ]; then
    # adjust the learning rate *=.5
    CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr .05 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-lr-.05.log
fi


echo "########## Adjusting the training epochs #############"

if [ ! -f logs/resnet-${depth}/resnet${depth}-cifar-${ct}-epochs-80.log ]; then
    # halfing the training epochs
    CUDA_VISIBLE_DEVICES=0 python main.py --epoch 80 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-epochs-80.log
fi 

if [ ! -f logs/resnet-${depth}/resnet${depth}-cifar-${ct}-epochs-320.log ]; then
    # doubling the training epochs
    CUDA_VISIBLE_DEVICES=0 python main.py --epoch 320 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-epochs-320.log
fi


echo "########## Adjusting the size of mini-batch #############"

if [ ! -f logs/resnet-${depth}/resnet${depth}-cifar-${ct}-mini-batch-256.log ]; then
    # mini-batch *= 2.0
    CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 256 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-mini-batch-256.log
fi

if [ ! -f logs/resnet-${depth}/resnet${depth}-cifar-${ct}-mini-batch-64.log ]; then
    # mini-batch *= .5
    CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 64 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-mini-batch-64.log
fi


if [ "$residual_block" -ne "$def_res"  ]; then
    echo " Adjusting the residual-blocks ${residual_block}"

    if [ ! -f logs/resnet-${depth}/resnet${depth}-cifar-${ct}-rb-${residual_block}.log ]; then
    # adjust the residual-blocks
        CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} -rb ${residual_block} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-rb-${residual_block}.log

    fi
fi

if [ "$skip_connections" -ne "$def_sc"  ]; then
    echo " Adjusting the skipping connections = ${skip_connections}"

    if [ ! -f logs/resnet-${depth}/resnet${depth}-cifar-${ct}-sc-${skip_connections}.log ]; then
    # adjust the skip-connections
        CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} -sc ${skip_connections} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-sc-${skip_connections}.log

    fi
fi

if [ "$train_data_size" -ne "$def_tds"  ]; then

    train_data_size=$(awk -v ts=$train_data_size 'BEGIN { print (ts / 10) }')

    echo " Adjusting the training data size connections = ${train_data_size} "

    if [ ! -f logs/resnet-${depth}/resnet${depth}-cifar-${ct}-tds-${train_data_size}.log ]; then
    #  adjust the skip-connections
        CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} -tds ${train_data_size} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-tds-${train_data_size}.log

    fi
fi

echo " finished "

