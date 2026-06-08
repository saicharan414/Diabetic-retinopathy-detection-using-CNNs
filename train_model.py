from sklearn.ensemble import RandomForestClassifier
import joblib

# Load preprocessed data
X_train = joblib.load('X_train.pkl')
y_train = joblib.load('y_train.pkl')

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'diabetes_model.pkl')