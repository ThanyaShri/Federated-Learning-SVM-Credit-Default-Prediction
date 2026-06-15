# Federated-Learning-SVM-Credit-Default-Prediction
Overview

This project implements a Federated Learning framework for credit default prediction using Support Vector Machine (SVM). The Flower framework is used to enable collaborative model training across multiple clients without sharing raw financial data.

Objectives:
Preserve data privacy during model training.
Train a global SVM model using Federated Averaging (FedAvg).
Predict credit default risk.
Evaluate model performance using classification metrics.

Technologies Used:
Python
Flower (FLWR)
NumPy
Pandas
Scikit-Learn
Matplotlib

Project Workflow:
Data preprocessing and normalization.
Distribution of data among multiple clients.
Local SVM training on each client.
Federated Averaging on the server.
Global model generation.
Performance evaluation.

Files:
server.py
Starts the federated server and performs FedAvg aggregation.

client.py
Loads client data, trains local SVM models, and communicates model parameters to the server.

evaluation.py
Evaluates model performance using:
Accuracy
Precision
Recall
F1 Score
ROC-AUC

Results:
The federated learning approach successfully trains an SVM classifier while maintaining data privacy and reducing the need for centralized data collection.

Future Enhancements:
Differential Privacy
Secure Aggregation
Deep Learning Models
Real-time Financial Analytics
