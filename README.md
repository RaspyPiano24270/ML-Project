# Severe Weather Prediction App: Rainfall Forecasting via LSTM

## Project Overview
This project develops a deep learning model to provide short-term, accurate predictions of severe weather events, such as rainfall amount or storm intensity, using historical atmospheric data. Accurate near-future forecasts are critical for flash flood warnings and disaster preparedness, improving upon traditional statistical forecasting methods.

## Problem Statement
Severe weather events often result from complex, non-linear temporal patterns in local atmospheric conditions that traditional models struggle to capture. Our goal is to leverage the temporal learning capabilities of a Long Short-Term Memory (LSTM) network to predict future weather metrics (rainfall amount or storm intensity class) based on sequences of recent historical observations.

## Dataset
* **Source:** Kaggle's Rainfall Prediction Dataset
* **Purpose:** Provides sequential historical weather data (temperature, pressure, humidity, wind speed, etc.).

## Model Architecture
* **Primary Model:** **Long Short-Term Memory (LSTM) Network.**
* **Rationale:** LSTMs are specialized Recurrent Neural Networks designed to effectively capture and remember long-term temporal dependencies (patterns over time) in sequential data like weather observations.
* **Task:** The model will take an input sequence (e.g., 24 hours of sensor readings) and output a prediction for the next hour.

## Evaluation Metrics

### Regression Task (Predicting Rainfall Amount)
* **Mean Squared Error (MSE):** Measures the average squared difference between the estimated values and the actual value.
* **Mean Absolute Error (MAE):** Measures the average magnitude of the errors, providing a more interpretable measure of error in the original units (e.g., millimeters of rain).

### Classification Task (Predicting Storm Intensity/Severity)
* **Accuracy:** Overall correct prediction percentage.
* **Precision, Recall, F1-score:** Essential metrics for evaluating severe events. They help distinguish between **false alarms** (low precision) and **missed events** (low recall).
* **Confusion Matrix:** A visualization required to fully understand false positives (false alarms) versus false negatives (missed events).

## Setup and Installation

To run this project, clone the repository and install the dependencies listed in `requirements.txt`.

```bash
git clone (https://github.com/RaspyPiano24270/ML-Project.git)
cd Severe-Weather-Prediction-App
pip install -r requirements.txt