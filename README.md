This README covers two different machine learning projects: House Price Classification and Regression and Cancer Detection using SVM and MLP Classifiers. Both projects include various techniques for data preprocessing, model training, evaluation, and visualization.

Table of Contents
House Price Classification and Regression

Data Loading and Initial Exploration

Data Preprocessing

K-Nearest Neighbors (KNN) Model

Visualization

Model Evaluation

K-Fold Cross-Validation

Data Visualization

Cancer Detection with SVM and MLP Classifiers

Data Preprocessing

Feature Selection

Model Training and Evaluation

Hyperparameter Tuning for MLP Classifier

Cross-validation

Model Comparison

Visualization

House Price Classification and Regression
This project performs house price classification using K-Nearest Neighbors (KNN) on a dataset containing various features of houses, such as square footage, number of bedrooms, number of bathrooms, and neighborhood quality. The house prices are classified into three categories: low, medium, and high.

Key Steps:
Data Loading and Initial Exploration

Load the dataset using pandas and perform basic exploratory data analysis (EDA).

Check for missing values and duplicates.

Preview the dataset using head(), describe(), and info().

Data Preprocessing

Categorize house prices into three classes: 'low', 'medium', and 'high' using pd.cut().

Split the dataset into training, validation, and test sets.

Scale the features using StandardScaler.

K-Nearest Neighbors (KNN) Model

Use GridSearchCV to tune hyperparameters like n_neighbors, metric, and weights.

Display the best parameters and cross-validation accuracy.

Visualization

Plot model complexity vs. performance to visualize the relationship between the number of neighbors and accuracy.

Compare different distance metrics (Euclidean vs. Manhattan) and weighting schemes (uniform vs. distance).

Add a red line to indicate the best value for n_neighbors.

Model Evaluation

Evaluate the best KNN model on the test set using a classification report and confusion matrix.

Display a heatmap of the confusion matrix.

K-Fold Cross-Validation

Use a pipeline with StandardScaler and the best KNN model for 5-fold cross-validation.

Report individual fold accuracies and the mean accuracy across all folds.

Data Visualization

Create a pair plot of selected features (e.g., Square Footage, Number of Bedrooms, Bathrooms) colored by Price_Class.

Use PCA for dimensionality reduction and generate a 2D scatter plot to visualize class separation.

Libraries Used:
pandas, numpy, matplotlib, seaborn, scikit-learn

Cancer Detection with SVM and MLP Classifiers
This project focuses on detecting breast cancer using machine learning techniques, particularly Support Vector Machines (SVM) and Multi-layer Perceptron (MLP) classifiers. The goal is to predict whether a tumor is malignant or benign based on biopsy data.

Key Steps:
Data Preprocessing

Load the dataset using pandas.

Drop irrelevant columns like 'Unnamed: 32' and 'id'.

Map the diagnosis labels: 'M' to 1 (malignant), 'B' to 0 (benign).

Handle missing values and verify dataset integrity.

Feature Selection

Calculate correlation of each feature with the target variable (diagnosis).

Select the most relevant features based on correlation.

Model Training and Evaluation

Train multiple SVM models with different kernels (linear, polynomial, RBF, and sigmoid).

Evaluate models using accuracy, confusion matrix, and classification report.

Hyperparameter Tuning for MLP Classifier

Tune the alpha hyperparameter using GridSearchCV.

Evaluate multiple MLP architectures with different neuron configurations and activation functions (Sigmoid, ReLU).

Cross-validation

Use cross-validation to evaluate model generalization.

Report mean cross-validation accuracy for each model.

Model Comparison

Compare the performance of SVM and MLP models.

Visualization

Plot loss curves for each MLP model to analyze convergence and overfitting/underfitting.

Libraries Used:
numpy, pandas, matplotlib, seaborn, scikit-learn, keras, scipy
