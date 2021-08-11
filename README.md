# Machine Learning

This repository provides introduction to machine learning models 

1. (regression) to predict math scores of a child from the income of the parents 
2. (classficiation) to predict whether a patient tests positive with Covid-19 from blood pressure, lung capacity, and body temperature 

using a simple, simulated synthetic data set. 

## Programming + Libraries:
I use [scikit-learn](https://scikit-learn.org/) library for Python programming language.

## Organization of the Repository:

| Jupyter Notebook | Content | Data set/Problem | 
|---|:---|:---|
|  [Linear Regressions](https://github.com/minyoungrho/MachineLearning/blob/main/classnotes/LinearRegression.ipynb)  | Linear Regressions  |   [Math Score](https://github.com/minyoungrho/MachineLearning/blob/main/data/scores_synth.csv) |
| [Nonlinear Regressions](https://github.com/minyoungrho/MachineLearning/blob/main/classnotes/NonLinearRegression.ipynb)  | Linear Regressions, Decision Trees, Random Forest, XGBoost  |  [Math Score](https://github.com/minyoungrho/MachineLearning/blob/main/data/scores_synth.csv) |
| [Classifications](https://github.com/minyoungrho/MachineLearning/blob/main/classnotes/Classfication.ipynb)  | Logistic Regressions, Support Vector Machine (SVM), k-Nearest Neighbor (KNN), XGBoost  |  [Covid-19](https://github.com/minyoungrho/MachineLearning/blob/main/data/synth_covid.csv) |   
| [Tune Hyperparameters](https://github.com/minyoungrho/MachineLearning/blob/main/classnotes/Hyperparameters.ipynb)  | GridSearch, RandomizedSearch  |  [Math Score](https://github.com/minyoungrho/MachineLearning/blob/main/data/scores_synth.csv) |  
|  [Dimensionality Reduction](https://github.com/minyoungrho/MachineLearning/blob/main/classnotes/HighDimensionalVariables.ipynb)  | Principal Component Analysis (PCA)  | [Iris](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) |   


## Description:
In this repository, we build machine learning predictive models using the following method 

1. Linear Regressions
2. Decision Trees
3. Random Forest
4. XGBoost
5. Logistic Regression
6. Support Vector Machine (SVM)
7. k-Nearest Neighbor (kNN)

and tune hyperparameters for each of the above method using the following algorithms

1. GridSearch
2. RandomizedSearch

and select the optimal model based on the following metrics calculated using cross-validation to quantify the quality of the prediction

1. Mean squared error
2. Absolute squared error
3. Accuracy score
4. area under the curve (AUC) for receiver operating characteristic (ROC)

The prediction from the selected models are visualized for each of the models.

Lastly, we also look into dimensionality reduction method called Principal Component Analysis (PCA).

