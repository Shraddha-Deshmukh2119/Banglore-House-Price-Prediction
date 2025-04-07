import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

def predict():
    # Generate synthetic dataset
    np.random.seed(42)
    data = {
        "emails_sent": np.random.randint(10, 500, 200),
        "streaming_hours": np.random.randint(1, 50, 200),
        "cloud_storage": np.random.randint(1, 500, 200),
        "device_usage": np.random.randint(1, 24, 200),
    }
    
    df = pd.DataFrame(data)
    df["carbon_footprint"] = (
        df["emails_sent"] * 0.3
        + df["streaming_hours"] * 36
        + df["cloud_storage"] * 2
        + df["device_usage"] * 50
    )

    # Split data
    X = df.drop(columns=["carbon_footprint"])
    y = df["carbon_footprint"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Save model
    with open("carbon_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    return predictions

# Example usage:

