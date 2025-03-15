import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def load_data():
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
        st.write("## Preview of Data")
        st.write(df.head())
        return df
    return None

def visualize_data(df):
    st.write("## Data Visualization")
    st.line_chart(df)

def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

def create_sequences(data, time_step=50):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def train_lstm(X_train, Y_train):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=10, batch_size=16, verbose=1)
    return model

def main():
    st.title("LSTM Time Series Analysis & Prediction")
    df = load_data()
    if df is not None:
        visualize_data(df)
        data, scaler = preprocess_data(df)
        
        time_step = st.slider("Select Time Step", 10, 100, 50)
        X, Y = create_sequences(data, time_step)
        train_size = int(len(X) * 0.8)
        X_train, Y_train = X[:train_size], Y[:train_size]
        X_test, Y_test = X[train_size:], Y[train_size:]
        
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        if st.button("Train LSTM Model"):
            model = train_lstm(X_train, Y_train)
            st.success("Model Trained Successfully!")
            
            Y_pred = model.predict(X_test)
            Y_pred = scaler.inverse_transform(Y_pred.reshape(-1, 1))
            Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))
            
            st.write("## Prediction vs Actual")
            fig, ax = plt.subplots()
            ax.plot(Y_test, label='Actual')
            ax.plot(Y_pred, label='Predicted')
            ax.legend()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
