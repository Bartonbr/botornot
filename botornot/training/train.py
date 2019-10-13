from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


class ModelTrainer:
    def __init__(self, model):
        self.model = model
        self.test_results = None
        self.train_results = None

    def train_model(self, features):
        bots = features[features['outcome'] == 1.0]
        not_bots = features[features['outcome'] == 0.0].head(len(bots))

        final_data = pd.concat([bots, not_bots])

        X = final_data[
            ['avg_bids',
             'unique_devices',
             'unique_ips',
             'avg_time_between_bids',
             'counterbid_time',
             'country']]
        y = final_data['outcome']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_train_pred = self.model.predict(X_train)

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

        return self.model


