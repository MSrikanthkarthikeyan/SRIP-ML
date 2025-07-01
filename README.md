# SRIP-ML
Summer Research Internship in Machine Learning Project - Efficient Predictive Maintenance using Multi-model ML along with FBprophet

# Efficient Predictive Maintenance in Smart Factories Using Multi-Model Machine Learning

🚀 A hybrid machine learning framework combining **real-time classification** and **time-series forecasting** to predict machinery failures before they happen in Industry 4.0 environments.

## 🧠 Overview

Predictive Maintenance (PdM) is the future of industrial operations—our framework is designed to reduce unplanned downtime, cut energy waste, and boost machine longevity. This project fuses structured sensor data with failure logs using a **multi-model ML architecture**, enhancing reliability across smart manufacturing systems.

## ✨ Key Features

- **Multi-Model Framework:** Combines Logistic Regression, KNN, Random Forest, Gradient Boosting, XGBoost, and Facebook Prophet.
- **Early Failure Detection:** Facebook Prophet achieved a 93.82% accuracy with 88.89% early detection rate.
- **Time-Series Forecasting:** Utilizes Prophet with external regressors and changepoint detection.
- **Real-Time Classification:** KNN, RF, and XGBoost achieved >98% accuracy in failure classification.
- **Hybrid Integration:** Smart ensemble combining classification with temporal forecasting for actionable insights.

## 🏗️ Architecture


graph TD;
    A[Sensor Data Collection] --> B[Preprocessing & Feature Engineering]
    B --> C[Classification Models]
    B --> D[FBProphet Time-Series Forecasting]
    C --> E[Hybrid Decision Logic]
    D --> E
    E --> F[Maintenance Alerts & Visualization]
🧪 Results Summary
Model	Accuracy
Logistic Regression	82.57%
K-Nearest Neighbors	98.94%
Random Forest	99.90%
Gradient Boosting	95.87%
XGBoost	99.63%
Prophet Forecast	93.82% (early detection rate: 88.89%)

📊 Dataset
Kaggle-based predictive maintenance dataset with telemetry and failure labels:
🔗 Dataset Link

📈 Use Cases
Smart Factories / IIoT Systems

Wind Turbine & Rotating Machinery Monitoring

Edge-enabled Predictive Systems

📌 Future Work
Autoencoders for unsupervised anomaly detection

Edge deployment & real-time pipeline integration

XAI (Explainable AI) for transparent maintenance insights

👨‍💻 Author
Srikanth Karthikeyan M
UG Researcher, VIT Chennai
📧 srikanth.karthikeyan2023@vitstudent.ac.in
