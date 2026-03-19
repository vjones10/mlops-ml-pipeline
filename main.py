from src.data_ingestion import load_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import feature_engineering
from src.model_training import train_model
from src.model_evaluation import evaluate_model

def run_pipeline():
    df = load_data()
    df = preprocess_data(df)
    X, y = feature_engineering(df)
    model, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    run_pipeline()