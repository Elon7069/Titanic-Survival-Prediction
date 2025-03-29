import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time

def load_preprocessed_data():
    """Load preprocessed data splits."""
    print("Loading preprocessed data...")
    data_path = os.path.join('models', 'processed_data.pkl')
    preprocessor_path = os.path.join('models', 'preprocessor.pkl')
    
    processed_data = joblib.load(data_path)
    preprocessor = joblib.load(preprocessor_path)
    
    return processed_data, preprocessor

def optimize_random_forest(X_train, y_train, preprocessor):
    """Optimize Random Forest hyperparameters using grid search."""
    print("\nOptimizing Random Forest hyperparameters...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=42))
    ])
    
    # Define parameter grid
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [5, 10, 20, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['auto', 'sqrt', 'log2']
    }
    
    # Create grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"Optimization complete in {end_time - start_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def optimize_gradient_boosting(X_train, y_train, preprocessor):
    """Optimize Gradient Boosting hyperparameters using grid search."""
    print("\nOptimizing Gradient Boosting hyperparameters...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', GradientBoostingClassifier(random_state=42))
    ])
    
    # Define parameter grid
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2],
        'model__subsample': [0.8, 1.0],
        'model__max_features': ['auto', 'sqrt']
    }
    
    # Create grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"Optimization complete in {end_time - start_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def create_ensemble_model(best_models):
    """Create an ensemble model from optimized models."""
    print("\nCreating ensemble model...")
    
    # Define estimators for voting classifier
    estimators = []
    for name, model in best_models.items():
        estimators.append((name, model))
    
    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=-1
    )
    
    return ensemble

def evaluate_model(model, X_test, y_test, model_name="Ensemble"):
    """Evaluate the trained model and plot ROC curve."""
    print(f"\nEvaluating {model_name} on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"AUC score: {auc_score:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Did Not Survive', 'Survived'],
                yticklabels=['Did Not Survive', 'Survived'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(os.path.join('models', f'confusion_matrix_{model_name}.png'))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='royalblue', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join('models', f'roc_curve_{model_name}.png'))
    
    return accuracy, auc_score, y_pred, y_proba

def main():
    """Main function to run advanced model training."""
    os.makedirs('models', exist_ok=True)
    
    # Load preprocessed data
    processed_data, preprocessor = load_preprocessed_data()
    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    
    # Optimize models
    rf_model, rf_params, rf_score = optimize_random_forest(X_train, y_train, preprocessor)
    gb_model, gb_params, gb_score = optimize_gradient_boosting(X_train, y_train, preprocessor)
    
    # Evaluate individual models
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "RandomForest")
    gb_metrics = evaluate_model(gb_model, X_test, y_test, "GradientBoosting")
    
    # Create and evaluate ensemble model
    best_models = {
        'RandomForest': rf_model.named_steps['model'],
        'GradientBoosting': gb_model.named_steps['model']
    }
    
    # Create ensemble pipeline
    ensemble = Pipeline([
        ('preprocessor', preprocessor),
        ('model', create_ensemble_model(best_models))
    ])
    
    # Train ensemble
    ensemble.fit(X_train, y_train)
    
    # Evaluate ensemble
    ensemble_metrics = evaluate_model(ensemble, X_test, y_test, "Ensemble")
    
    # Save best model
    best_model_name = max([("RandomForest", rf_metrics[0]), 
                          ("GradientBoosting", gb_metrics[0]), 
                          ("Ensemble", ensemble_metrics[0])], 
                         key=lambda x: x[1])[0]
    
    if best_model_name == "RandomForest":
        best_model = rf_model
        best_accuracy = rf_metrics[0]
    elif best_model_name == "GradientBoosting":
        best_model = gb_model
        best_accuracy = gb_metrics[0]
    else:
        best_model = ensemble
        best_accuracy = ensemble_metrics[0]
    
    # Save model
    model_path = os.path.join('models', f'{best_model_name}_advanced.pkl')
    joblib.dump(best_model, model_path)
    
    # Save model info
    model_info = {
        'model_name': best_model_name,
        'accuracy': best_accuracy,
        'parameters': best_model.get_params()
    }
    joblib.dump(model_info, os.path.join('models', 'advanced_model_info.pkl'))
    
    print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    print(f"Model saved to {model_path}")
    print("\nAdvanced model training complete!")

if __name__ == "__main__":
    main() 