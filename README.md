# Experiments on CIFAR datasets with PyTorch implementation of ResNet.Hyperparameter adjustments

This is a survey version forked from https://github.com/junyuseu/pytorch-cifar-models

## Introduction
Investigate with hyperparameters adjustments  on state-of-the-art CNN models in cifar dataset with PyTorch, now including:

1.[ResNet](https://arxiv.org/abs/1512.03385v1)

## Requirements:software
Requirements for [PyTorch](http://pytorch.org/)

## Requirements:hardware
For most experiments, one or two K40(~11G of memory) gpus is enough cause PyTorch is very memory efficient. However,
to train DenseNet on cifar(10 or 100), you need at least 4 K40 gpus.

## Usage
1. Clone this repository

```
git clone https://github.com/Tenfleques/resnet-survey.git
```

In this project, the network structure is defined in the models folder, the script ```gen_mean_std.py``` is used to calculate
the mean and standard deviation value of the dataset.

1. run ./run.sh -d &lt;depth = 20|56|110 default=20&gt; -c &lt;cifar dataset = 10|100 default=10&gt;

This will train and dump the history in the logs folder.
    ```
        logs/

         |_____ resnet-20/

                |_____ charts/ 

                ...

         |_____ resnet-56/

                |_____ charts/

                ...

         |_____ resnet-110/

                |_____ charts/

                ...
    ```
After training, the training log will be recorded in the .log file, the best model(on the test set) 
will be stored in the fdir.

**Note**:For first training, cifar10 or cifar100 dataset will be downloaded, so make sure your comuter is online.
Otherwise, download the datasets and decompress them and put them in the ```data``` folder.

2. run ./investigate-models.sh 

    -d &lt;depth = 20|56|110 default=20&gt; 

    -c &lt;cifar dataset = 10|100 default=10&gt; 

    -p &lt;hyper-params variants separated by commas whose results to investigate = -lr-.01|-lr-.02|-lr-.2|-lr-.5|-lr-1.0|-epochs-80|-epochs-320|-mini-batch-64|-mini-batch-256 default='-lr-.01,-lr-.02,-lr-.2,-lr-.5,-lr-1.0,-epochs-80,-epochs-320,-mini-batch-64,-mini-batch-256'&gt;

    -m &lt;metrics to compare = loss_val_mean|acc_val_mean default='loss_val_mean,acc_val_mean'&gt;


3. Test

```
CUDA_VISIBLE_DEVICES=0 python main.py -e --resume=fdir/model_best.pth.tar
```

5. CIFAR100

The default setting in the code is for cifar10, to train with cifar100, you need specify it explicitly in the code.

```
model = resnet20_cifar(num_classes=100)
```

## Results


### ResNet cifar-10

hyper params|layers|error(%)|prec(%)|
:---:|:---:|:---:|:---:

### ResNet cifar-100

hyper params|layers|error(%)|prec(%)|
:---:|:---:|:---:|:---:



# References:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.

[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.

[3] S. Zagoruyko and N. Komodakis. Wide residual networks. In BMVC, 2016.

[4] S. Xie, G. Ross, P. Dollar, Z. Tu and K. He Aggregated residual transformations for deep neural networks. In CVPR, 2017

[5] H. Gao, Z. Liu, L. Maaten and K. Weinberger. Densely connected convolutional networks. In CVPR, 2017
# resnet-survey
