from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import joblib


class ModelTrainer:
    def __init__(self, model):
        self.model = model
        self.test_results = None
        self.train_results = None

    def train_model(self, features, features_y):
        x_train, x_test, y_train, y_test = train_test_split(features, features_y, test_size=0.33, random_state=42)

        self.model.fit(x_train, y_train)

        y_pred = self.model.predict(x_test)
        y_train_pred = self.model.predict(x_train)

        self.test_results = {
            'Accuracy:': accuracy_score(y_test, y_pred),
            'F1:': f1_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred)}

        self.train_results = {
            'Accuracy:': accuracy_score(y_train, y_train_pred),
            'F1:': f1_score(y_train, y_train_pred),
            'Recall': recall_score(y_train, y_train_pred),
            'Precision': precision_score(y_train, y_train_pred)}

    def persist_model(self, path):
        joblib.dump(self.model.best_estimator_, path)
