# Causality-based-event-sequence-modelling
Implementation of the Casual based Neural Hawkes model as a course project for the Models of Sequential Data course.

# Dataset
For experiments, we use the data from the original implementation of Neural Hawkes as also synthetic data. 
It can be downloaded via [this link](https://www.dropbox.com/s/bu02lcs3wcmh5zk/data.zip?dl=0).
Each dataset is contained in a separate folder with corresponding name. 
Data files are separated into the following splits:
* **Train** : Training the model
* **Dev** : Tuning hyper parameters
* **Test** : Final test and numbers in the paper

In this repo we store all data in [data folder](data/)
All obtaning results are avaliable via [this link](https://www.dropbox.com/s/uyldl8lzr9djxnh/artefacts.zip?dl=0)

# Existing solutions
We use tick library as a source of implementations for Granger Causality for Hawkes process estimation.
We also rely on tick in the process of synthetic dataset generation.

[Neural Hawkes](https://github.com/Hongrui24/NeuralHawkesPytorch) is existing PyTorch implementation of Neural Hawkes based on the original implementation from the authors of the approach([nh](https://github.com/xiao03/nh)), which is also included. 
We will use this implementation as a startign point of our own reimplementation. 
It provides simple and understandable solution. 
Moreover, it provides useful datasets for experiments with the model. 
This solution code can be found in [this folder](related_implementations/)

# Repository structure
The code for our approach is stored in [causal_nh/](causal_nh/).
Our implementation can be installed from the [setup.py](setup.py) file as a package.
It requires Pytorch and tick libraries to be installed.

Our package contains two main modules: granger causality and model.
Granger causality module contains a function to estimate matrices A and W from the given dataset. 
An example of its usage on our synthetic data can be found in the [Causality_estimation](causal_nh/notebooks/Causality_estimation.ipynb). 

The model module contains modified versions of Neural Hawkes models. 
The code for training and testing the model is contained in the root of causal\_nh, while the example of its usage can be found in [Train_NeuralHawkes_main](causal_nh/notebooks/Train_NeuralHawkes_main.ipynb).
Additionally, module with utils contains useful functions for plotting and synthetic dataset generation, with more detail on this functions in [Obtaining_synthetic_data](causal_nh/notebooks/Obtaining_synthetic_data.ipynb). 

Data folder contains all possible datasets on which the model can be run, including the original Neural Hawkes datasets and our synthetic ones.
In the folder with related implementations we store the original implementations of Neural Hawkes, on which we base our code.

# Repo structure
``` 
├── README.md
├── artefacts
├── causal_nh
│   ├── __init__.py
│   ├── granger_causality
│   │   ├── __init__.py
│   │   └── granger_causality_graph.py
│   ├── model
│   │   ├── ContTimeLSTM_Cell.py
│   │   ├── NeuralHawkes.py
│   │   └── __init__.py
│   ├── notebooks
│   │   ├── Causality_estimation.ipynb
│   │   ├── Data_Exploration.ipynb
│   │   ├── Obtaining_synthetic_data.ipynb
│   │   ├── Train_NeuralHawkes_main.ipynb
│   │   └── dataset_summary.csv
│   ├── test.py
│   ├── tests
│   │   ├── test_CausalNH.py
│   │   └── test_NH.py
│   ├── train.py
│   └── utils.py
├── data
│   ├── NeuralHawkesData
│   │   ├── data_bookorder
│   │   ├── data_conttime
│   │   ├── data_hawkes
│   │   ├── data_hawkesinhib
│   │   ├── data_meme
│   │   ├── data_mimic
│   │   ├── data_missing
│   │   ├── data_retweet
│   │   ├── data_retweet_sampled
│   │   └── data_so
│   ├── README.md
│   ├── data_synth_10_events_small
│   ├── data_synth_2_events_small
│   ├── data_synth_3_events_small
│   └── data_synth_5_events_small
│       ├── A.pkl
│       ├── dev.pkl
│       ├── test.pkl
│       └── train.pkl
├── related_implementations
│   ├── NeuralHawkesPytorch
│   ├── README.md
│   └── nh_master
└── setup.py

``` 
