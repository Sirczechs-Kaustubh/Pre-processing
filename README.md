# Practical 4

## Introduction

For this week, we covered common pre-processing techniques applied during the machine learning (ML) pipeline. For the practical, you will put some of these into practice. You will work with the <ins> Pima Indian Diabetes <ins> and the aim will be to compare the effects of pre-processing on two learners: k-nearest neighbor (kNN) and multilayer perceptron (MLP). The pre-processing that you will apply will include scaling, under-sampling the majority class and over-sampling the minority class. 


## Procedure

1. Clone the repo to github

!git clone https://github.com/Dr-M-ELBA/Practical_4.git

2. import the libraries

3. ```Python

   import pandas as pd
   import numpy as np
   import sys
  
   # pre-processing libraries
   from sklearn.preprocessing import StandardScaler
   from sklearn.pipeline import make_pipeline

   # ML training libraries
   from sklearn.model_selection import train_test_split
   from sklearn.neural_network import MLPClassifier
   from sklearn.neighbors import KNeighborsClassifier

   # Evaluation libraries
   from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

   # Class imbalance libraries
   
   
