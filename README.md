Model Evaluation and Hyperparameter Tuning


Objective<br/>

The goal of this project is to evaluate and compare three models: Decision Tree, Random Forest, and K-Nearest Neighbors for a binary classification task.<br/>
You will also tune the hyperparameters of these models using Grid Search, perform 5-fold cross-validation, and visualize performance metrics using boxplots.<br/>
Additionally, you will plot ROC curves for model comparison on a testing set.<br/>

Dataset<br/>

We will use a standard binary classification dataset such as the Breast Cancer dataset from sklearn:<br/>

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()<br/>.

STEPS<br/>

Step 1: Data Preprocessing<br/>

1. Load the dataset and split it into training (80%) and testing (20%) sets.<br/>

2. Standardize numerical features. Which Scaler do you need to use?<br/>

Step 2: Model Building<br/>

1. Decision Tree Model:<br/>
- Build a Decision Tree model.
- Tune hyperparameters such as max_depth, min_samples_split, and min_samples_leaf using Grid Search.
  
2. Random Forest Model:<br/>
- Build a Random Forest model.
- Tune hyperparameters such as max_depth, min_samples_split, n_estimators, and min_samples_leaf using Grid Search.

3. K-Nearest Neighbors Model:<br/>
- Build a K-Nearest Neighbors (KNN) model.
- Tune hyperparameters such as n_neighbors, weights, and p (distance metric) using Grid Search.<br/>

Below are the grids for GridSearchCV:
param_grid_dt = {
'max_depth': [3, 5, 10, None],
'min_samples_split': [2, 10, 20],
'min_samples_leaf': [1, 5, 10]
}<br/>

param_grid_rf = {
'n_estimators': [50, 100, 200],
'max_depth': [None, 10, 20],
'min_samples_split': [2, 10],
'min_samples_leaf': [1, 5]
}<br/>

param_grid_knn = {
'n_neighbors': [3, 5, 7, 9],
'weights': ['uniform', 'distance'],
'p': [1, 2]
}<br/>

Decision Tree (DT) Grid:
- max_depth: Controls the maximum depth of the tree. Options: 3, 5, 10, or no limit (None).
- min_samples_split: Minimumnumberofsamplesrequiredtosplit an internal node. Options: 2, 10, 20.
- min_samples_leaf: Minimum number of samples required to be at a leaf node. Options: 1, 5, 10.<br/>

Random Forest (RF) Grid:
- n_estimators: Number of trees in the forest. Options: 50, 100, 200.
- max_depth: Maximumdepthofeachtree. Options: None (nolimit), 10, 20.
- min_samples_split: Minimumnumberofsamplesrequiredtosplit a node. Options: 2, 10.
- min_samples_leaf: Minimum number of samples required at a leaf node. Options: 1, 5.

K-Nearest Neighbors (KNN) Grid:
- n_neighbors: Number of neighbors to use for knn. Options: 3, 5, 7, 9.
- weights: Weight function used in prediction. Options: uniform, distance.
- p: Power parameter for the Minkowski metric. Options: 1 (Manhattan distance), 2 (Euclidean distance).<br/>

Step 3: Model Evaluation
1. Perform 5-fold Cross-Validation for each model.
2. Evaluate the models using the following metrics:
- Accuracy
- Precision
- Recall
- Specificity
- F1-Score
- Area Under the Curve (AUC)<br/>

3. Use boxplots to visualize the distribution of the metrics across the 5 cross-validation folds.<br/>

Step 4: ROC Curve<br/>

1. For each model, plot the ROC curve using the testing set and compare the models based on the AUC metric.<br/>

Step 5: Model Selection<br/>
1. Compare the models based on the cross-validation metrics and ROC curves.<br/>

2. Select the best-performing model and justify your choice. Discuss how hyperparameter tuning affected the model’s performance.<br/>
