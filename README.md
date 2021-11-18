# Causality-based-event-sequence-modelling
Implementation of the Casual based Neural Hawkes model as a course project for the Models of Sequential Data course.

# Dataset
For experiments we use the data from the original implementation of Neural Hawkes. 
It can be downloaded via [this link](https://drive.google.com/drive/folders/0BwqmV0EcoUc8UklIR1BKV25YR1U?resourcekey=0-OrlU87jyc1m-dVMmY5aC4w&usp=sharing).
Each dataset is contained in a separate folder with corresponding name. 
Data files are separated into the following splits:
* **Train** : Training the model
* **Dev** : Tuning hyper parameters
* **Test** : Final test and numbers in the paper
* **(Test1)** : Future test

In this repo we store all data in [data folder](data/)

# Existing solutions
[THAP: A matlab Tool for HAwkes Processes](https://github.com/HongtengXu/Hawkes-Process-Toolkit) provides functionality for Hawkes Process analysis in MatLab language.
We use this library as inspiration for Granger Causality for Hawkes process implimentation.
This solution code can be found in [this folder](related_implementations/Hawkes-Process-Toolkit-master)

[Neural Hawkes](https://github.com/xiao03/nh) is existing PyTorch implementation of Neural Hawkes from original authors of the approach. 
We will use this implementation as a startign point of our own reimplementation. 
It provides simple and understandable solution. 
Moreover, it provides useful datasets for experiments with the model. 
This solution code can be found in [this folder](related_implementations/nh-master)

# Our approach
The code for our approach is stored in [causal_nh/ folder](causal_nh/).
We divided it into two modules: grander causality and the modified Neural Hawkes model.
Examples of the code usage can be found in [notebooks/ folder](causal_nh/notebooks/)


# Repo structure
``` 
├── causal_nh
│   ├── granger_causality
│   ├── model
│   ├── notebooks
│   └── __init__.py
├── data
│   ├── data_hawkesinhib
│   └── README.md
├── related_implementations
│   ├── Hawkes-Process-Toolkit-master
│   ├── nh-master
│   └── README.md
└── README.md

``` 
