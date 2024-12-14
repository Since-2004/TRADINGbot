import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Step 1: Load and Preprocess Data
def load_data(file_path):
    """Load stock data from a CSV file."""
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

def preprocess_data(data):
    """Preprocess the data: feature engineering and scaling."""
    # Feature Engineering
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = compute_rsi(data['Close'], window=14)
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    # Handle missing values
    data = data.dropna(subset=['SMA_10', 'SMA_50', 'RSI', 'Target'])  # Drop rows required for model training

    if data.empty:
        raise ValueError("Dataset is empty after preprocessing. Ensure your dataset has enough rows.")

    # Scaling
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'RSI']
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])

    return data, features, scaler

def compute_rsi(series, window):
    """Compute the Relative Strength Index (RSI)."""
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Step 2: Train-Test Split
def split_data(data, features):
    """Split the dataset into training and testing sets."""
    X = data[features]
    y = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

# Step 3: Train the Model
def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 4: Evaluate the Model
def evaluate_model(model, X_test, y_test):
    """Evaluate the model using classification metrics."""
    y_pred = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 5: Save the Model and Scaler
def save_model(model, scaler, model_path, scaler_path):
    """Save the trained model and scaler to disk."""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

# Main Pipeline
def main():
    # File path to stock data
    file_path = 'AAPL_historical_data.csv'  # Replace with your dataset file path

    try:
        # Load and preprocess data
        data = load_data(file_path)
        print("Loaded data preview:")
        print(data.head())
        data, features, scaler = preprocess_data(data)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = split_data(data, features)

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        evaluate_model(model, X_test, y_test)

        # Save the model and scaler
        save_model(model, scaler, 'stock_model.pkl', 'scaler.pkl')
        print("Model and scaler saved successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
