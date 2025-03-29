# Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic using various classification algorithms and feature engineering techniques.

## Overview

This project uses the famous Titanic dataset to build and train machine learning models that predict which passengers survived the Titanic disaster. The goal is to demonstrate how effective feature engineering and ensemble modeling can improve prediction accuracy.

## Features

- **Data Preprocessing**: Handles missing values and transforms categorical features
- **Feature Engineering**: Extracts titles from names, creates family size, fare per person features, and more
- **Multiple Models**: Implements Random Forest, Gradient Boosting, SVC, and Logistic Regression
- **Ensemble Model**: Combines multiple models for better accuracy (83.2%)
- **Feature Importance Analysis**: Visualizes which features contribute most to survival predictions
- **Interactive UI**: Streamlit-based web interface for making predictions and exploring the dataset

## Project Structure

```
titanic-survival-prediction/
│
├── data/                  # Dataset files
│   └── titanic.csv        # Titanic dataset
│
├── models/                # Saved model files and visualizations
│   ├── Ensemble_advanced.pkl
│   ├── enhanced_feature_importance.png
│   ├── model_accuracy_comparison.png
│   └── ...
│
├── src/                   # Source code
│   ├── app.py             # Streamlit web application
│   ├── preprocess.py      # Data preprocessing functions
│   ├── enhanced_feature_importance.py # Feature importance visualization
│   └── ensemble_model.py  # Ensemble model implementation
│
└── README.md              # Project documentation
```

## Installation and Usage

1. Clone the repository:
   ```
   git clone https://github.com/Elon7069/Titanic-Survival-Prediction.git
   cd Titanic-Survival-Prediction
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```
   streamlit run src/app.py
   ```

4. Access the application in your browser at `http://localhost:8501`

## Models and Performance

The project implements several machine learning models:

| Model | Accuracy |
|-------|----------|
| Random Forest | 83.2% |
| Gradient Boosting | 79.9% |
| SVC | 83.8% |
| Logistic Regression | 82.1% |
| Ensemble | 83.2% |

## Key Insights

- **Sex** and **Title** are the most important features for predicting survival
- **Ticket Number** and **Age** also play significant roles in prediction
- Social class (represented by **Pclass**) affects survival chances
- Family relationships impact survival rates

## Future Improvements

- Implement hyperparameter tuning for each base model
- Add more advanced feature engineering techniques
- Explore deep learning approaches
- Incorporate more data visualization options

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The dataset is from the Kaggle Titanic competition
- Thanks to the scikit-learn, pandas, and Streamlit communities 