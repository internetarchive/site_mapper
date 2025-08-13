"""
Train, tune, evaluate, and save the Random Forest model.

Inputs:
- results/training_data_processed.csv: Preprocessed dataset

Outputs:
- results/model/random_forest_model.joblib: Trained model
- results/model/model_info.json: Feature names, feature importances, model params, evaluation metrics

Flow:
- Load processed data and select label (prefers 'label_fixed' if present, else 'label_contextual')
- Train/test split (80/20)
- Hyperparameter tuning via GridSearchCV (StratifiedKFold)
- Cross-validation metrics and test evaluation (classification report, confusion matrix, ROC AUC)
- Save model and metadata for inference

Usage:
  uv run python src/site_mapper/train_random_forest.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json
from pathlib import Path

def load_preprocessed_data(csv_path):
    """
    Load the preprocessed training data from CSV and split into features/target.

    Args:
        csv_path: Path-like string to results/training_data_processed.csv

    Returns:
        X: pandas DataFrame of feature columns
        y: pandas Series of target labels (bool/int)
        feature_cols: list of feature column names used in training
    """
    
    print(f"Loading preprocessed data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Separate features and target - use FIXED labels
    target_col = 'label_fixed'
    if target_col not in df.columns:
        # Fallback to original if fixed labels not available
        target_col = 'label_contextual'
        print("Using original labels - run fix_training_labels.py first for better results")
    
    feature_cols = [col for col in df.columns if col not in ['label_contextual', 'label_simple', 'label_fixed', 'label_reason', 'url', 'link_text', 'source_page']]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(df)}")
    print(f"Positive class ratio: {y.mean():.2%}")
    
    return X, y, feature_cols

def analyze_class_balance(y):
    """
    Analyze class distribution and recommend whether to use class weighting.

    Args:
        y: pandas Series of target labels

    Returns:
        use_class_weight: bool indicating if class_weight='balanced' should be used
    """
    class_counts = np.bincount(y.astype(int))
    class_ratio = class_counts[1] / class_counts[0] if class_counts[0] > 0 else 1
    
    print(f"\nClass Distribution:")
    print(f"  Bad links (0): {class_counts[0]:,} ({class_counts[0]/len(y):.1%})")
    print(f"  Good links (1): {class_counts[1]:,} ({class_counts[1]/len(y):.1%})")
    print(f"  Class ratio (good/bad): {class_ratio:.2f}")
    
    # Recommend class weighting if imbalanced
    if class_ratio < 0.5 or class_ratio > 2.0:
        print("Imbalanced classes detected - will use class_weight='balanced'")
        return True
    else:
        print("Classes reasonably balanced")
        return False

def hyperparameter_tuning(X_train, y_train, use_class_weight=False):
    """
    Perform hyperparameter tuning via GridSearchCV and return the best estimator.

    Args:
        X_train: pandas DataFrame of training features
        y_train: pandas Series of training labels
        use_class_weight: whether to enable class weighting

    Returns:
        best_estimator_: trained RandomForestClassifier with best found params
    """
    print("\nPerforming hyperparameter tuning...")
    

    param_grid = {
        'n_estimators': [100, 200, 300],  # Number of trees
        'max_depth': [10, 15, 20, None],  # Tree depth
        'min_samples_split': [2, 5, 10],  # Min samples to split
        'min_samples_leaf': [1, 2, 4],    # Min samples in leaf
        'max_features': ['sqrt', 'log2', 0.3]  # Features per split
    }
    
    # Base model
    rf_base = RandomForestClassifier(
        random_state=42,
        class_weight='balanced' if use_class_weight else None,
        n_jobs=-1 
    )
    

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        rf_base, 
        param_grid, 
        cv=cv,
        scoring='f1',  # F1 score balances precision/recall
        n_jobs=-1,
        verbose=1
    )
    
    print("Running grid search (this may take a few minutes)...")
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate a trained model on the test set and report metrics.

    Args:
        model: fitted RandomForestClassifier
        X_test: pandas DataFrame test features
        y_test: pandas Series test labels
        feature_names: list of feature column names used for importance

    Returns:
        dict with keys: 'auc_score', 'feature_importance' (DataFrame), 'classification_report' (dict)
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Bad Link', 'Good Link']))
    

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    

    auc_score = roc_auc_score(y_test, y_prob)
    print(f"\nROC AUC Score: {auc_score:.4f}")
    

    print("\n" + "="*30)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*30)
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
        print(f"{i+1:2d}. {row['feature']:25s} {row['importance']:.4f}")
    
    # Check for potential overfitting
    train_score = model.score(X_test, y_test)  # Using test as proxy since we don't have separate validation
    print(f"\nTest Accuracy: {train_score:.4f}")
    
    return {
        'auc_score': auc_score,
        'feature_importance': feature_importance,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def cross_validation_analysis(model, X, y):
    """
    Run stratified k-fold cross-validation with several scoring metrics.

    Args:
        model: fitted RandomForestClassifier
        X: pandas DataFrame features
        y: pandas Series labels
    """
    print("\n" + "="*30)
    print("CROSS-VALIDATION ANALYSIS")
    print("="*30)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Multiple scoring metrics
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    for metric in scoring_metrics:
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
        print(f"{metric:10s}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

def save_model_and_results(model, results, feature_names, output_dir):
    """
    Persist the trained model and evaluation artifacts to disk.

    Args:
        model: fitted RandomForestClassifier
        results: dict returned by evaluate_model
        feature_names: list of feature column names
        output_dir: directory path where artifacts will be written
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = output_dir / "random_forest_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save feature names and importance
    model_info = {
        'feature_names': feature_names,
        'feature_importance': results['feature_importance'].to_dict('records'),
        'model_params': model.get_params(),
        'evaluation_metrics': results['classification_report']
    }
    
    info_path = output_dir / "model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    print(f"Model info saved to: {info_path}")

def main():
    """
    Main training pipeline:
      1) Load processed data
      2) Analyze class balance
      3) Train/test split
      4) Hyperparameter tuning
      5) Cross-validation
      6) Final evaluation and save artifacts
    """
    print("="*60)
    print("RANDOM FOREST TRAINING FOR CRAWLER LINK FILTERING")
    print("="*60)
    
    # 1. Load preprocessed data
    data_path = Path("results/training_data_processed.csv")
    X, y, feature_names = load_preprocessed_data(data_path)
    
    # 2. Analyze class balance
    use_class_weight = analyze_class_balance(y)
    
    # 3. Split data (80/20 split)
    print(f"\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # 4. Hyperparameter tuning
    best_model = hyperparameter_tuning(X_train, y_train, use_class_weight)
    
    # 5. Cross-validation analysis
    cross_validation_analysis(best_model, X_train, y_train)
    
    # 6. Final evaluation on test set
    results = evaluate_model(best_model, X_test, y_test, feature_names)
    
    # 7. Save model and results
    save_model_and_results(best_model, results, feature_names, "results/model")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"Final ROC AUC: {results['auc_score']:.4f}")

if __name__ == "__main__":
    main()