from risk_predictor import RiskScorePredictor

if __name__ == "__main__":
    predictor = RiskScorePredictor()
    X_train, X_test, y_train, y_test = predictor.load_and_prepare_data("diabetes.csv", "Outcome")
    predictor.train(X_train, y_train)
    results = predictor.evaluate(X_test, y_test)

    print("Model Evaluation Results:")
    print(f"Accuracy: {results['accuracy']:.2f}")
    print("Confusion Matrix:")
    print(results["confusion_matrix"])
    print("Classification Report:")
    print(results["classification_report"])
