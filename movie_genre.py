# ===============================
# MOVIE GENRE CLASSIFICATION PROJECT
# ===============================

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# ===============================
# 1. CREATE DUMMY DATA (if no dataset)
# ===============================
if not os.path.exists("movies.csv"):
    print("Creating dummy dataset 'movies.csv'...")

    dummy_data = {
        'plot': [
            "A group of friends are attacked by zombies in a forest.",
            "Two lovers meet and fall in love in Paris.",
            "A detective solves a mysterious murder case.",
            "A hero saves the world with action and explosions.",
            "A haunted house terrorizes a family.",
            "A romantic story of love and heartbreak."
        ] * 20,
        'genre': ['Horror', 'Romance', 'Mystery', 'Action', 'Horror', 'Romance'] * 20
    }

    pd.DataFrame(dummy_data).to_csv("movies.csv", index=False)

# ===============================
# 2. LOAD DATA
# ===============================
df = pd.read_csv("movies.csv")

# ===============================
# 3. FEATURE ENGINEERING
# ===============================
df['plot_length'] = df['plot'].apply(len)
df['word_count'] = df['plot'].apply(lambda x: len(str(x).split()))

X = df[['plot', 'plot_length', 'word_count']]
y = df['genre']

# ===============================
# 4. TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 5. PREPROCESSING PIPELINE
# ===============================
text_feature = 'plot'
numeric_features = ['plot_length', 'word_count']

preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True
        ), text_feature),
        
        ('num', StandardScaler(), numeric_features)
    ]
)

# ===============================
# 6. MODEL 1: LOGISTIC REGRESSION + GRID SEARCH
# ===============================
pipeline_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=500, class_weight='balanced'))
])

param_grid = {
    'model__C': [0.1, 1, 10],
    'preprocessor__text__max_df': [0.8, 1.0]
}

print("🔍 Training Logistic Regression with GridSearch...")
grid_search = GridSearchCV(
    pipeline_lr,
    param_grid,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_lr_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# ===============================
# 7. MODEL 2: SVM
# ===============================
pipeline_svm = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearSVC())
])

print("\n🔍 Training SVM...")
pipeline_svm.fit(X_train, y_train)

# ===============================
# 8. EVALUATION
# ===============================
print("\n===== Logistic Regression Results =====")
y_pred_lr = best_lr_model.predict(X_test)
print(classification_report(y_test, y_pred_lr, zero_division=0))

print("\n===== SVM Results =====")
y_pred_svm = pipeline_svm.predict(X_test)
print(classification_report(y_test, y_pred_svm, zero_division=0))

# ===============================
# 9. CONFUSION MATRIX
# ===============================
print("\n📊 Confusion Matrix (Best Model)")
ConfusionMatrixDisplay.from_estimator(best_lr_model, X_test, y_test)
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# ===============================
# 10. SAVE MODEL
# ===============================
joblib.dump(best_lr_model, "movie_genre_model.pkl")
print("\n✅ Model saved as movie_genre_model.pkl")

# ===============================
# 11. CUSTOM PREDICTION FUNCTION
# ===============================
def predict_genre(plot_text):
    sample = pd.DataFrame({
        'plot': [plot_text],
        'plot_length': [len(plot_text)],
        'word_count': [len(plot_text.split())]
    })
    
    prediction = best_lr_model.predict(sample)[0]
    return prediction

# ===============================
# 12. TEST PREDICTION
# ===============================
test_plot = "A haunted house with ghosts and scary events"
print("\n🎬 Sample Prediction:")
print("Plot:", test_plot)
print("Predicted Genre:", predict_genre(test_plot))
