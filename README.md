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
   import matplotlib.pyplot as plt
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
   from imblearn.under_sampling import RandomUnderSampler
   from imblearn.over_sampling import RandomOverSampler, SMOTE ```

4. Make sure the libraries are installed. On colab, you can execute a cell with !pip install followed by the code. For example: !pip install imbalanced-learn

5. Once you've imported the relevant libraries, try the following
   a. Upload the dataset and check for any missing values.Feel free to perform any other exploratory data analysis (EDA).
   b. Assign the input features to X and the target to y
   c. Compare the results of MLP when the dataset is unscaled vs scaled. Compare the accuracy, f1 score when the minority class is set as the positive class, and the MCC.
   d. Compare the results of kNN when the dataset is unscaled vs scaled. Compare the accuracy, f1 score when the minority class is set as the positive class, and the MCC.

   Note, make sure to use the make_pipeline function to avoid data leakage. For example:

   ```Python

   scaled_mlp_pipeline = make_pipeline(StandardScaler(), MLPClassifier(random_state=42)) ```
   
