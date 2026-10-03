"""
model.py
RandomForest Classifier for F1 starting tyre prediction.
Includes training, hyperparameter tuning (GridSearchCV), and persistence.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
try:
    from src.data_loader import get_feature_names, COMPOUND_LABELS
except ModuleNotFoundError:
    from data_loader import get_feature_names, COMPOUND_LABELS


RANDOM_STATE = 21
MODEL_PATH   = "outputs/randomForest_model.pkl"


def split_data(df: pd.DataFrame, test_size: float = 0.20):
    """Split into 80/20 train-test sets (stratified by compound)."""
    features = get_feature_names()
    X = df[features]
    y = df["StartingCompound"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,          # Preserve class balance in both splits
    )
    print(f"[Model] Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test


from sklearn.ensemble import RandomForestClassifier

def train(X_train, y_train, tune_hyperparams: bool = True):
    """
    Train a Random Forest Classifier.

    If tune_hyperparams=True, runs GridSearchCV to find the best
    n_estimators, max_depth, and min_samples_split.
    """
    if tune_hyperparams:
        print("[Model] Running GridSearchCV for hyperparameter tuning …")
        param_grid = {
            "n_estimators":      [200, 300],
            "max_depth":         [6, 8, 10],
            "min_samples_split": [4, 6, 8],
            "min_samples_leaf":  [1, 2],
        }
        base_clf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            base_clf,
            param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        clf = grid_search.best_estimator_
        print(f"[Model] Best params: {grid_search.best_params_}")
        print(f"[Model] Best CV accuracy: {grid_search.best_score_:.4f}")
    else:
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

    # 5-fold cross-validation on training data
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
    print(f"[Model] 5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return clf


def save_model(clf: RandomForestClassifier, path: str = MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    # Also save to decision_tree_model.pkl for backward compatibility with README examples
    alt_path = "outputs/decision_tree_model.pkl"
    if path != alt_path:
        with open(alt_path, "wb") as f:
            pickle.dump(clf, f)
    print(f"[Model] Saved to '{path}' and '{alt_path}'")



def load_model(path: str = MODEL_PATH) -> RandomForestClassifier:
    with open(path, "rb") as f:
        clf = pickle.load(f)
    print(f"[Model] Loaded from '{path}'")
    return clf


def predict_single(clf: RandomForestClassifier,
                   grid_pos: int,
                   track_temp: float,
                   air_temp: float,
                   rain_prob: int,
                   team: int) -> dict:
    """
    Predict the starting compound for a single driver scenario.

    Returns a dict with the predicted label and class probabilities.
    """
    sample = pd.DataFrame([{
        "GridPosition": grid_pos,
        "TrackTemp":    track_temp,
        "AirTemp":      air_temp,
        "TempDelta":    round(track_temp - air_temp, 1),
        "IsTop10":      int(grid_pos <= 10),
        "Team":         team,
    }])

    pred_class = clf.predict(sample)[0]
    proba      = clf.predict_proba(sample)[0]

    return {
        "predicted_compound":  COMPOUND_LABELS[pred_class],
        "confidence":          f"{max(proba) * 100:.1f}%",
        "probabilities": {
            COMPOUND_LABELS[i]: f"{p * 100:.1f}%"
            for i, p in enumerate(proba)
        },
    }

