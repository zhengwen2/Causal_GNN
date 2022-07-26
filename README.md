# Causal GNN

This repository is the implementation of causal GNN. To solve the problem of generalization in GNN. The SCM model we define is as follows：
<div align=center><img width="150" height="150" src="https://github.com/zhengwen2/Causal_GNN/blob/master/scm.png"/></div>
z is the intention variable, x and xn represent the characteristics of the central node and its neighbor nodes respectively, sn is structure information.
The implementation process is shown in the following figure：
<div align=center><img width="300" height="200" src="https://github.com/zhengwen2/Causal_GNN/blob/master/model.png"/></div>

## Requirements

- python 3.9
- pytorch 1.10

## Usage

To train the model(s), run this command:

```train
python train.py 
```
