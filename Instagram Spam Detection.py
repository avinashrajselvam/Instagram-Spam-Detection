import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Step 1: Load Data
train = pd.read_csv("Train.csv")
test = pd.read_csv("test.csv")

# Step 2: EDA
print("Initial data info:")
print(train.info())
print("\nTarget distribution:\n", train['fake'].value_counts())

# Optional: Visualizations
sns.countplot(x='fake', data=train)
plt.title("Fake vs Genuine Accounts")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(train.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Step 3: Data Preprocessing
X = train.drop('fake', axis=1)
y = train['fake']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test.drop('fake', axis=1))

# Step 4: Model Building
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluation
y_pred = model.predict(X_val)
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 6: Feature Importance
importances = model.feature_importances_
features = train.columns.drop('fake')

feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
print("\nFeature Importances:\n", feat_imp)

feat_imp.plot(kind='bar')
plt.title("Feature Importance")
plt.ylabel("Importance Score")
plt.show()

# Step 7: Predict on Test Data (Optional)
test_preds = model.predict(test_scaled)
print("\nSample Predictions on Test Set:\n", test_preds[:10])
