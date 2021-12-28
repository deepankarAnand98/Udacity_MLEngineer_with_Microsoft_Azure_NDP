# Optimizing an ML Pipeline in Azure

## Overview

This project is part of the Udacity Azure ML Nanodegree.
This project builds and optimizes an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is compared to an Azure AutoML run.

## Summary
In this project, we are working with bank telemarketing data. It consists of 39 attributes explaining the success of telemarketing campaigns and a target variable, whether or not the customer subscribed to the term deposit. 
We aim to create a classification model that maps attributes to the target variable and predicts that a new customer will apply for term deposit with the bank based on given features.

The best performing model was a VotingEnsemble which yielded the accuracy of 91.70%.  
![AutoML Metrics](https://github.com/deepankarAnand98/Udacity_MLEngineer_with_Microsoft_Azure_NDP/blob/main/Optimizing_ML_Pipeline/images/automl_metrics.PNG)
![AutoML Accuracy](https://github.com/deepankarAnand98/Udacity_MLEngineer_with_Microsoft_Azure_NDP/blob/main/Optimizing_ML_Pipeline/images/automl_accuracy.PNG)
The accuracy produced by hyperparameter tuning using Hyperdrive on Logistic Regression is 91.138%.
![Hyperdrive Accuracy](https://github.com/deepankarAnand98/Udacity_MLEngineer_with_Microsoft_Azure_NDP/blob/main/Optimizing_ML_Pipeline/images/hyperdrive_accuracy.PNG)

Below are the top 10 features as obtained using Global Feature Importance
![Feature Importance](https://github.com/deepankarAnand98/Udacity_MLEngineer_with_Microsoft_Azure_NDP/blob/main/Optimizing_ML_Pipeline/images/feature_importance.PNG)

The top three features are 
1. Duration - last contact duration
2. nr_employed - total number of employees
3. emp_var_rate - employment variation rate 

### AutoML Normalized Confusion Matrix
![Normalized Confusion Matrix](./images/confusion_matrix_automl.PNG)
The confusion matrix by automl shows that our TN ratio is ~96%. Given the features, our model correctly predicted 96 out of 100 times that the person won't subscribe to the term deposit. This will help the band analyze and understand the customers who did not convert and change their marketing approach. 
However, our model is only ~57.78% successful in predicting those customers who actually took Term deposits. Our True Positive Ratio is ~58, and False Positive Ratio is 42%. This indicates that our model is less efficient in identifying a set of features of the successfully converted customer. In other words, the characteristics of the people who subscribed to the term deposit are almost similar to those who didn't subscribe, which is confusing in our model.

## Scikit-learn Pipeline
In the hyperparameter tuning part, we've created a python script that runs logistic regression using scikit-learn. We'll use the bank marketing data and apply hyperparameter tuning using Hyperdrive and identify the most optimal set of parameters. 

We have used Random parameter sampling. The Hyperdrive will randomly select the hyperparameters from the list of parameters. The benefit of this approach is that it speeds up the model training without much compromising on the model's accuracy.

For early termination, we have used BanditPolicy. We have set 0.2 as our slack factor. As soon as our primary metric is not within the slack factor of the best performing run, the current run will terminate.

## AutoML
For applying AutoML, first, we create a tabular dataset using the same bank marketing dataset. Afterwards, we define the configuration for our AutoML model. We specify the type of ml task (*classification* in our case), the primary metric to be used, specify attributes and the target variable, and our compute instance.

### AutoML Parameters 
1. `experiment_timeout_hours`: the maximum amount of time before the experiment terminates.
2. `task`: kind of ml task we want to run. Choices: (classification,regression,forecasting).
3.  `primary_metric`: the decision metric, based on the type of task, which the automl model will optimize.
4.   `training_data`: dataset on which we want to run AutoML experiment.
5.   `label_column_name`: the target column in the provided dataset.
6.   `n_cross_validations`: the number of cross validation each time model is trained.
7.   `compute_target`: compute target which is used to run the AutoML experiment.
8.   `max_concurrent_iterations`: the maximum number of iterations allowed to run in parallel.

## Pipeline comparison
Both Hyperdrive and AutoML have different architecture because each has to perform specific tasks. The control flow is mentioned below.

### Hyperdrive Pipeline
1. Initializing a parameter search space.  
We have used 
2. Defining sampling method.  
3. Specify the primary metric with a goal(MAXIMIZE, MINIMIZE etc.).  
4. Define an early stopping policy.  

### AutoML Scikit-Learn Pipeline
1. Creating a Tabular Dataset
2. Cleaning the dataset using local functions in `train.py` file
3. Set up AutomML Configuration
   * Define ML Task
   * Declare Primary Metric
   * Set Compute Target
   * Set Training Data
   * Specify target column

The hyperparameter tuning using Hyperdrive was completed in 17m 46s, whereas the AutoML experiment took almost 27m to complete. AutoML trains multiple models on the data, where each model training equals one child run. In the case of hyperparameter tuning, one child runs on a specific set of hyperparameters. Since the number of a child runs exceeded that of the Hyperdrive experiment, the AutoML experiment took more time.

## Future work
1. **Feature selection to help increase the True Positive Ratio of the best model.**   
   We need to include features specific to the indivuals who subscibed to the term deposit. These features must be majorly absent in those customers who did not subsribe to the term deposit.
2. **Test model for fairness**  
   Since the bank market its schemes to people of all ethnicity, we must ensure that our ML model does not product predictions which are baised towards certain class section of people in the dataset. Hence test for fairness is crucial.
3. **Deploy the model on the web**  
   Deploying the model as a web service will make it possible for others to use it over the internet.

## Proof of cluster clean up
![cluster_delete](https://github.com/deepankarAnand98/Udacity_MLEngineer_with_Microsoft_Azure_NDP/blob/main/Optimizing_ML_Pipeline/images/delete_compute_cluster.PNG)
