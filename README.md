# MIR-Project

The following code comprehends the two implementations of "Towards Deep Learnt Intra-frame Instrument Classification", Gor, S. & Gusó, E., UPF-SMC, MIR 2019 Project. It classifies frames of 700 ms in four classes (Vocals, Drums, Bass and Other) as corresponds to MUSDB18 Dataset. 

The scripts in SVM approach correspond to our two baselines: a Scikit-learn implementation with a:
* Support Vector Machine 
* Boosted Decision Tree (XGBoost)

The scripts in DNN approach correspond to a Pytorch implementation of :
* ResNet-18
* ResNet-152
* Plain-CNN
* The encoder part of a U-net
* Multilayer Perceptron

For more details, run:
```
python dnn_based/main.py -h
```

## Requeriments

### Packages

Install the needed packages creating a conda environment using the provided "conda_req.txt" file:
```
conda create -n new environment python=3.7.2 --file conda_req.txt 
```

Install musdb package for loading the dataset:
```
pip install musdb
```

### Cuda

These requirements install Cuda 9.0 pytorch version. If you use another version of Cuda, manually overwrite pytorch installation.
ResNet-152 model requires 10GB of VRAM.

## Dataset

The used dataset is MUSDB18 from SigSep Datasets and it must be located inside MIR-Project folder. Dataset can be obtained here:  https://sigsep.github.io/datasets/musdb.html#tools


