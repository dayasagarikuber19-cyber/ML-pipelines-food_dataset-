This project is formulated as a supervised multiclass classification problem. Each food item represents a unique class, and the model learns from labeled nutritional data to classify inputs into one of many possible food categories.

Why Multiclass Classification?

The target variable (Food Name) contains multiple categories

Each nutritional profile corresponds to exactly one food item

Ensures strict control in diet planning scenarios

Business Objective
:
To automate food selection in health-focused applications by accurately classifying foods based on nutritional profiles, reducing manual errors and improving adherence to diet plans.

Approach: 

Data Understanding & EDA: Explore data distribution, class balance, and visualize feature trends

Data Preprocessing: Handle missing values, remove duplicates, treat outliers, and scale numeric features

Feature Engineering & Selection: Reduce dimensionality using PCA, select most relevant features to improve model efficiency

Model Training: Train and compare multiple classifiers including Logistic Regression, Decision Tree, Random Forest, KNN, SVM, and Gradient Boosting

Model Evaluation: Evaluate using accuracy, precision, recall, F1-score, and confusion matrix to understand class-wise performance.
Models Used:
Logistic Regression
Decision Tree
Random Forest
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Gradient Boosting

Evaluation Metrics:

Accuracy – overall model correctness,

Precision – class-wise positive prediction quality,

Recall – ability to identify all positive cases,

F1-Score – balance between precision and recall,

Confusion Matrix – detailed class-wise performance visualization.


