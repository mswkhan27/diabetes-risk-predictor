import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class RiskScorePredictor:
    """
    Predicts risk of developing a chronic condition based on patient health data.

    Uses Random Forest Classifier for binary classification (high-risk vs low-risk).

    Attributes:
    -----------
    model : sklearn.ensemble.RandomForestClassifier
        The classifier used for prediction.

    scaler : sklearn.preprocessing.StandardScaler
        Used for feature scaling of input variables.
    """

    def __init__(self, n_estimators=100, random_state=42):
        self.model =  RandomForestClassifier(class_weight="balanced", random_state=random_state)
        self.random_state = random_state
        self.scaler = StandardScaler()

    def load_and_prepare_data(self, file_path: str, target_column: str):
        """
        Loads and preprocesses healthcare data from CSV file.

        Parameters:
        -----------
        file_path : str
            Path to CSV file containing health data.
        target_column : str
            Column name representing the classification target.

        Returns:
        --------
        X_train_scaled, X_test_scaled, y_train, y_test : preprocessed data splits
        """
        df = pd.read_csv(file_path)
        df = df.dropna()

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train(self, X_train, y_train):
        # Random Forest with hyperparameter tuning

        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False]
        }

        grid_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_grid,
            n_iter=15,
            cv=5,
            verbose=1,
            scoring="accuracy",
            random_state=self.random_state,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

    def predict(self, X):
        """
        Predicts classification for input features.

        Parameters:
        -----------
        X : array-like
            Scaled feature matrix.

        Returns:
        --------
        array : Predicted labels.
        """
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model using accuracy and classification metrics.

        Parameters:
        -----------
        X_test : array-like
            Scaled test feature matrix.
        y_test : array-like
            True labels.

        Returns:
        --------
        dict : Accuracy, confusion matrix, and classification report.
        """
        y_pred = self.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=False)
        }
