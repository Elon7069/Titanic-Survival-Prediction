import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def load_preprocessed_data():
    """Load preprocessed data splits."""
    print("Loading preprocessed data...")
    data_path = os.path.join('models', 'processed_data.pkl')
    preprocessor_path = os.path.join('models', 'preprocessor.pkl')
    
    processed_data = joblib.load(data_path)
    preprocessor = joblib.load(preprocessor_path)
    
    return processed_data, preprocessor

def train_model(X_train, y_train, preprocessor):
    """Train and tune multiple models, then select the best one."""
    print("Training models...")
    
    # Create model candidates
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'SVC': SVC(random_state=42, probability=True)
    }
    
    # Parameters for GridSearchCV
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'LogisticRegression': {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'saga']
        },
        'SVC': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf']
        }
    }
    
    best_models = {}
    model_scores = {}
    
    # Create pipelines and perform grid search for each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            pipeline,
            param_grid={f'model__{param}': values for param, values in param_grids[model_name].items()},
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Save best model
        best_models[model_name] = grid_search.best_estimator_
        model_scores[model_name] = grid_search.best_score_
        
        print(f"{model_name} best parameters: {grid_search.best_params_}")
        print(f"{model_name} CV accuracy: {grid_search.best_score_:.4f}")
    
    # Select best model
    best_model_name = max(model_scores, key=model_scores.get)
    best_model = best_models[best_model_name]
    
    print(f"\nBest model: {best_model_name} with CV accuracy: {model_scores[best_model_name]:.4f}")
    
    return best_model, best_model_name

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    print("\nEvaluating model on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    
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
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join('models', 'confusion_matrix.png'))
    
    return accuracy, y_pred

def save_model(model, model_name):
    """Save the trained model."""
    print("\nSaving model...")
    model_path = os.path.join('models', f'{model_name}.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def main():
    """Main function to run model training."""
    # Load preprocessed data
    processed_data, preprocessor = load_preprocessed_data()
    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    
    # Train model
    best_model, best_model_name = train_model(X_train, y_train, preprocessor)
    
    # Evaluate model
    accuracy, y_pred = evaluate_model(best_model, X_test, y_test)
    
    # Save model
    save_model(best_model, best_model_name)
    
    # Save model info
    model_info = {
        'model_name': best_model_name,
        'accuracy': accuracy,
        'parameters': best_model.get_params()
    }
    joblib.dump(model_info, os.path.join('models', 'model_info.pkl'))
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main() 