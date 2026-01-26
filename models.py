
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb


df = pd.read_csv("food_dataset_cleaned.csv")  


X = df.drop('Food_Name', axis=1)
y = df['Food_Name']
le = LabelEncoder() # encoding
y_encoded = le.fit_transform(y)
scaler = StandardScaler() # feature scaling
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print("Original Features:", X.shape[1])
print("Reduced Features after PCA:", X_pca.shape[1])


X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"\n{name}")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

# Model_Comparison

plt.figure(figsize=(10,5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.xticks(rotation=45)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

# Confusion _ matrix 
best_model = RandomForestClassifier(n_estimators=200)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#Real_Time_Prediction
def predict_food(calories, protein, carbs, fat, sugar):
    input_data = np.array([[calories, protein, carbs, fat, sugar]])
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)

    prediction = best_model.predict(input_pca)
    return le.inverse_transform(prediction)[0]

# Example
predict_food(
    calories=250,
    protein=30,
    carbs=10,
    fat=5,
    sugar=2
)



