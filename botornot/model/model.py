from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def bot_or_not():
    parameters = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 100, None],
        'min_samples_split': [2, 4, 10],
        'min_samples_leaf': [1, 2, 4, 10],
        'class_weight': ['balanced']}
    return GridSearchCV(RandomForestClassifier(), parameters, cv=10, verbose=2)
