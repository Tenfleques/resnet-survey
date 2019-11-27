# Experiments on CIFAR datasets with PyTorch implementation of ResNet.Hyperparameter adjustments

This is a survey version forked from https://github.com/junyuseu/pytorch-cifar-models

## Introduction
Investigate with hyperparameters adjustments  on state-of-the-art CNN models in cifar dataset with PyTorch:

1.[ResNet](https://arxiv.org/abs/1512.03385v1)

## Requirements:software
Requirements for [PyTorch](http://pytorch.org/)

## Requirements:hardware
For most experiments, one or two K40(~11G of memory) gpus is enough cause PyTorch is very memory efficient.

## Usage
1. Clone this repository

```
git clone https://github.com/Tenfleques/resnet-survey.git
```

In this project, the network structure is defined in the models folder, the script ```gen_mean_std.py``` is used to calculate
the mean and standard deviation value of the dataset.

1. run 
```
./run.sh -d <depth = 20|50|56|62|110 default=20> -c <cifar dataset = 10|100 default=10>
```

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

2. run 
```
./investigate-models.sh 

    -d <depth = 20|56|110 default=20> 

    -c <cifar dataset = 10|100 default=10> 

    -p <hyper-params variants separated by commas whose results to investigate, default='lr.01,lr.02,lr.2,lr.5,lr1.0,e80,e320,mb64,mb256,rb0,rb2,rb3,sc1,sc3,td5,td2'>

    -m <metrics to compare = loss_val_mean|acc_val_mean default='loss_val_mean,acc_val_mean'>

    e.g 

    ./investigate-models.sh  -d 20,56 -c 10,100 -p lr.01,sc1 -m loss_val_mean
```


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

layers|epochs|mini-batch|learning-rate|residual-blocks|skip-connections|train-data size|precision|train avg_loss|train val_loss|test avg_loss|test val_loss
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
110|160|64|0.1|-1|-1|1.0|94.03|0.214399|0.213416|0.404220|0.404220
110|160|128|.2|-1|-1|1.0|94.00|0.239813|0.236899|0.464519|0.464519
56|160|128|.2|-1|-1|1.0|93.86|0.217774|0.216203|0.445965|0.445965
110|160|128|0.1|-1|-1|1.0|93.64|0.185127|0.183633|0.430926|0.430926
110|320|128|0.1|-1|-1|1.0|93.45|0.091158|0.090429|0.371125|0.371125
56|160|64|0.1|-1|-1|1.0|93.38|0.223451|0.222284|0.420656|0.420656
56|160|128|0.1|-1|-1|1.0|93.23|0.168009|0.168057|0.411599|0.411599
56|320|128|0.1|-1|-1|1.0|93.05|0.087030|0.087088|0.366807|0.366807
56|160|128|.5|-1|-1|1.0|92.94|0.368743|0.367106|0.607396|0.607396
56|160|256|0.1|-1|-1|1.0|92.59|0.174516|0.171087|0.463293|0.463293
110|160|128|.5|-1|-1|1.0|92.19|0.603813|0.597245|0.824913|0.824913
110|160|256|0.1|-1|-1|1.0|92.14|0.206344|0.201615|0.505260|0.505260
20|160|128|.2|-1|-1|1.0|92.10|0.228000|0.228609|0.464842|0.464842
20|160|64|0.1|-1|-1|1.0|91.88|0.234175|0.233673|0.427115|0.427115
20|160|128|.5|-1|-1|1.0|91.71|0.321341|0.322167|0.542035|0.542035
20|320|128|0.1|-1|-1|1.0|91.71|0.105918|0.106472|0.401427|0.401427
20|160|128|0.1|-1|-1|1.0|91.37|0.190934|0.190424|0.444061|0.444061
20|160|256|0.1|-1|-1|1.0|91.18|0.174758|0.174342|0.451299|0.451299
56|160|128|.02|-1|-1|1.0|91.16|0.150284|0.149980|0.469984|0.469984
110|160|128|.02|-1|-1|1.0|90.74|0.154860|0.154194|0.515237|0.515237
110|160|128|.01|-1|-1|1.0|90.40|0.178159|0.176723|0.542187|0.542187
56|160|128|.01|-1|-1|1.0|90.11|0.175090|0.175444|0.542940|0.542940
20|160|128|.02|-1|-1|1.0|89.91|0.180205|0.181132|0.466320|0.466320
110|160|128|1.0|-1|-1|1.0|89.40|0.667961|0.651716|0.952603|0.952603
20|160|128|.01|-1|-1|1.0|89.32|0.219818|0.218970|0.489794|0.489794
56|160|128|1.0|-1|-1|1.0|89.16|0.610519|0.605692|0.881119|0.881119
20|160|128|1.0|-1|-1|1.0|88.21|0.661367|0.660047|0.931684|0.931684
20|80|128|0.1|-1|-1|1.0|87.33|0.346406|0.345614|0.539951|0.539951
110|80|128|0.1|-1|-1|1.0|87.14|0.353701|0.352471|0.539776|0.539776
56|80|128|0.1|-1|-1|1.0|85.17|0.316144|0.315000|0.503280|0.503280


### ResNet cifar-100

layers|epochs|mini-batch|learning-rate|residual-blocks|skip-connections|train-data size|precision|train avg_loss|train val_loss|test avg_loss|test val_loss
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
110|160|64|0.1|-1|-1|1.0|72.34|0.714020|0.721438|1.614977|1.614977
110|160|128|0.1|-1|-1|1.0|72.14|0.607312|0.613392|1.718227|1.718227
110|160|128|.2|-1|-1|1.0|71.97|0.752711|0.762594|1.747861|1.747861
110|320|128|0.1|-1|-1|1.0|71.38|0.312704|0.316509|1.628282|1.628282
110|160|128|.5|-1|-1|1.0|71.18|1.162117|1.171058|1.920744|1.920744
56|160|64|0.1|-1|-1|1.0|70.80|0.766951|0.774677|1.590795|1.590795
56|160|128|.2|-1|-1|1.0|70.69|0.741960|0.753595|1.656276|1.656276
56|160|128|.5|-1|-1|1.0|70.67|1.037177|1.048340|1.823473|1.823473
56|160|128|0.1|-1|-1|1.0|70.46|0.623194|0.632670|1.628008|1.628008
56|320|128|0.1|-1|-1|1.0|70.30|0.360338|0.365390|1.624094|1.624094
110|160|256|0.1|-1|-1|1.0|68.96|0.611237|0.611140|1.838781|1.838781
20|160|64|0.1|-1|-1|1.0|68.62|1.003315|1.009754|1.580425|1.580425
110|160|128|.02|-1|-1|1.0|68.54|0.574578|0.580584|1.799948|1.799948
56|160|256|0.1|-1|-1|1.0|68.20|0.614713|0.616616|1.767400|1.767400
20|160|128|.2|-1|-1|1.0|67.56|0.976413|0.986122|1.646577|1.646577
20|160|128|0.1|-1|-1|1.0|67.36|0.900988|0.911926|1.607058|1.607058
20|160|128|.5|-1|-1|1.0|67.28|1.196526|1.208645|1.772918|1.772918
56|160|128|1.0|-1|-1|1.0|67.07|1.473065|1.482423|2.106433|2.106433
56|160|128|.02|-1|-1|1.0|66.54|0.798209|0.803042|1.818794|1.818794
20|320|128|0.1|-1|-1|1.0|66.50|0.619131|0.626158|1.545605|1.545605
110|160|128|1.0|-1|-1|1.0|65.91|1.674770|1.680427|2.240599|2.240599
20|160|128|1.0|-1|-1|1.0|65.88|1.517612|1.527368|2.062015|2.062015
20|160|256|0.1|-1|-1|1.0|65.61|0.880605|0.884464|1.670274|1.670274
110|160|128|.01|-1|-1|1.0|64.95|0.671925|0.677554|1.942655|1.942655
56|160|128|.01|-1|-1|1.0|64.89|0.741417|0.745706|1.806625|1.806625
20|160|128|.02|-1|-1|1.0|64.61|0.931732|0.937005|1.638080|1.638080
20|160|128|.01|-1|-1|1.0|63.31|1.079412|1.083510|1.696018|1.696018
110|80|128|0.1|-1|-1|1.0|60.02|1.167705|1.183752|1.907923|1.907923
20|80|128|0.1|-1|-1|1.0|57.90|1.318984|1.331280|1.848817|1.848817
56|80|128|0.1|-1|-1|1.0|57.65|1.166492|1.180245|1.815063|1.815063




# References:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.

[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.

[3] S. Zagoruyko and N. Komodakis. Wide residual networks. In BMVC, 2016.

[4] S. Xie, G. Ross, P. Dollar, Z. Tu and K. He Aggregated residual transformations for deep neural networks. In CVPR, 2017

[5] H. Gao, Z. Liu, L. Maaten and K. Weinberger. Densely connected convolutional networks. In CVPR, 2017
# resnet-survey
