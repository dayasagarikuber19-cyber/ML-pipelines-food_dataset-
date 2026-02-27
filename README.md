This project develops a multi-class classification model that predicts the exact food name based on nutritional attributes such as calories, protein, carbohydrates, fats, and sugar. The objective is to build a strict diet-compliant system that selects one precise food option based on nutritional requirements.

Business Motivation:
      In structured diet-planning systems, users often specify nutritional goals such as high-protein or low-sugar meals. Instead of loosely recommending multiple options, this system ensures strict classification by predicting one exact food that aligns with predefined dietary constraints.

Problem Type:
      This is a supervised multi-class classification problem where the input features are numerical nutritional metrics and the target variable is the food name.

Dataset Description:
      The dataset consists of tabular nutritional information where each row represents a unique food item. Nutritional metrics serve as predictive features, and the food name is the label to be predicted.

Feature Selection Strategy:
      Only core nutritional features were selected: Calories, Protein, Carbohydrates, Fats, and Sugar. These were chosen because they directly align with diet planning logic and reduce noise from irrelevant categorical attributes.

Data Cleaning and Preprocessing:
      The preprocessing pipeline included removing duplicates, handling missing values, validating numeric data types, and standardizing numerical features to ensure model stability and fair comparison across algorithms.

Feature Scaling:
      StandardScaler was applied to normalize feature magnitudes. This step was especially important for distance-based models like KNN and SVM, which are sensitive to feature scale.

Target Encoding:
      Since the target variable was categorical (food name), Label Encoding was applied to convert food names into numerical labels suitable for classification algorithms.

Train-Test Split Strategy:
      The dataset was split into training and testing sets using an 80-20 ratio with a fixed random state to ensure reproducibility and unbiased evaluation.

The following models were evaluated:

Logistic Regression. 1

Decision Tree . 2

Random Forest .3

K-Nearest Neighbors.4

Support Vector Machine.5

Gradient Boosting.6

This provided a comparison between linear, tree-based, ensemble, and distance-based approaches.


Model performance was evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Cross-validation accuracy.

F1-score was emphasized due to the multi-class nature of the dataset.

Handling Class Distribution:
Class distribution was analyzed before training. Since macro and weighted F1-scores were closely aligned, severe imbalance was not observed, and additional resampling techniques were not required.

 Best Performing Model:

    Random Forest achieved the highest accuracy and macro F1-score. Its ensemble learning approach improved generalization and reduced overfitting compared to single decision trees.

Feature Importance Analysis:

    Feature importance from the Random Forest model indicated that protein and fat were strong differentiating factors among food categories, validating domain logic in diet planning.

Overfitting Check:

Cross-validation was performed to ensure the selected model generalizes well. Comparable training and validation performance confirmed minimal overfitting.

 Real-Time Applicability:

The system can be integrated into smart diet applications where users input nutritional targets and receive a predicted food option instantly. This supports automation and strict diet adherence.

SUMMARY:
   “I developed a multi-class classification system that predicts exact food names based on nutritional metrics to support strict diet planning applications. I performed preprocessing, feature scaling, model benchmarking, and evaluation using accuracy and F1-score. Random Forest achieved the best performance, and cross-validation confirmed strong generalization. The system demonstrates real-time applicability for automated meal planning and health tracking.”
