import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load your dataset
df = pd.read_csv("predictive_maintenance_dataset.csv")  # Replace with your CSV path

# Drop non-feature columns
X = df.drop(['date', 'device', 'failure'], axis=1)
y = df['failure']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Calculate scale_pos_weight for XGBoost
scale_pos_weight = len(y_train_resampled[y_train_resampled == 0]) / len(y_train_resampled[y_train_resampled == 1])

# Define models with balancing strategies
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "Extra Trees": ExtraTreesClassifier(class_weight='balanced', random_state=42),
    "KNN": KNeighborsClassifier(),  # KNN doesn't use class weights
    "XGBoost": XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Train and evaluate
for name, model in models.items():
    print(f"\n🔧 Model: {name}")
    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    recall_failure = cm[1,1] / (cm[1,1] + cm[1,0]) * 100 if (cm[1,1] + cm[1,0]) else 0

    print(f"✅ Accuracy: {acc:.4f}")
    print(f"📈 Detection % (Recall of failures): {recall_failure:.2f}%")
    print(f"📊 Confusion Matrix:\n{cm}")
    print(f"🧾 Classification Report:\n{classification_report(y_test, y_pred)}")
