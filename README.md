# Online Payment Fraud Detection using Machine Learning

This project implements an **Online Payment Fraud Detection System** using Machine Learning (ML) algorithms. The system leverages multiple ML models to detect fraudulent transactions based on a dataset, using Python, Flask, and various ML libraries. The objective is to build a reliable fraud detection system that can classify online payments as fraudulent or non-fraudulent.

## Features

- **Fraud Detection**: Detects fraudulent transactions based on historical transaction data.
- **Comparative Study**: A comparison of various machine learning models, including:
  - Decision Tree
  - Logistic Regression
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Gradient Boosting Classifier
  - Naive Bayes
  - Neural Network
  - Support Vector Machine (SVM)
- **Model Evaluation**: Evaluation of models using key metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

## Tech Stack

- **Backend**: Python
- **Framework**: Flask (for web API)
- **Machine Learning Libraries**:
  - `Scikit-learn` for model implementation
  - `XGBoost` and `LightGBM` for enhanced models
  - `TensorFlow` and `Keras` for deep learning models
  - `Pandas`, `NumPy`, and `Matplotlib` for data manipulation and visualization
  - `Seaborn` for data visualization

## Installation

###1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/online-payment-fraud-detection.git
   cd online-payment-fraud-detection.
```
###2. Create a virtual environment (optional but recommended):
   ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  ```
###3.Running the Flask App:
```bash
python app.py
```

Running the Model
The system automatically trains multiple ML models and compares their performance.
You can input transaction data to get predictions for fraud detection.


Comparative Study
The system uses the following machine learning algorithms and compares their performance:

Decision Tree: A non-linear model that splits the data into branches to make predictions.
Logistic Regression: A statistical model used for binary classification.
Random Forest: An ensemble learning method using multiple decision trees.
K-Nearest Neighbors (KNN): A non-parametric model based on finding the most similar data points.
Gradient Boosting Classifier: An ensemble model that builds trees sequentially, focusing on errors.
Naive Bayes: A probabilistic classifier based on Bayes’ theorem.
Neural Network: A deep learning model designed to recognize patterns in large datasets.
Support Vector Machine (SVM): A model that finds the hyperplane that best separates classes in the feature space.


Evaluation Metrics
Accuracy: The ratio of correctly predicted transactions to the total transactions.
Precision: The ratio of true positive predictions to all positive predictions.
Recall: The ratio of true positive predictions to all actual fraudulent transactions.
F1-Score: The harmonic mean of precision and recall.
ROC-AUC: Measures the area under the receiver operating characteristic curve.


Conclusion
This project provides a comparative analysis of multiple machine learning algorithms for detecting online payment fraud, helping identify the most effective model based on various evaluation metrics.

