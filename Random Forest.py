import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv('train-balanced-sarcasm.csv', usecols=['label', 'parent_comment', 'score'])

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)       # Remove numbers
    return text

df['cleaned_text'] = df['parent_comment'].apply(clean_text)

# Split data
X = df[['cleaned_text', 'score']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create processing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        # Text features (applied to 'cleaned_text' column)
        ('text', TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), 'cleaned_text'),
        
        # Numerical feature (applied to 'score' column)
        ('num', StandardScaler(), ['score'])
    ])

# Full working pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        max_depth=15,
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model (optional)
import joblib
joblib.dump(model, 'sarcasm_detection_rf.pkl')