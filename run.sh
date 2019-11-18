#!/bin/bash

OPTIND=1

depth=20
ct=10

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
    esac
done


# run resnet-${depth} with default params
echo "########## resnet-${depth} dataset-${ct} epochs 160, batch-size 128, learning rate 0.1, ground paramaters #########"

CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}.log

echo "########## Adjusting the learning rate #############"

# adjust the learning rate *=10
CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr 1.0 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-lr-1.0.log

# adjust the learning rate *=5
CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr .5 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-lr-.5.log

# adjust the learning rate *=2
CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr .2 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-lr-.2.log

# adjust the learning rate *=.1
CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr .01 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-lr-.01.log

# adjust the learning rate *=.2
CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr .02 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-lr-.02.log

# adjust the learning rate *=.5
CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr .05 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-lr-.05.log


echo "########## Adjusting the training epochs #############"

# halfing the training epochs

CUDA_VISIBLE_DEVICES=0 python main.py --epoch 80 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-epochs-80.log


# doubling the training epochs
CUDA_VISIBLE_DEVICES=0 python main.py --epoch 320 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-epochs-320.log


echo "########## Adjusting the size of mini-batch #############"

# mini-batch *= 2.0
CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 256 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-mini-batch-256.log

# mini-batch *= .5
CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 64 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct ${ct} -d ${depth} > logs/resnet-${depth}/resnet${depth}-cifar-${ct}-mini-batch-64.log


