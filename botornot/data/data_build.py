import pandas as pd
import botornot.data.feature_engineering as fe


def build_training_data(raw_train):
    bots = raw_train[raw_train['outcome'] == 1.0]
    not_bots = raw_train[raw_train['outcome'] == 0.0].sample(n=len(bots), random_state=42)

    balanced_data = pd.concat([bots, not_bots])

    return build_data(balanced_data)


def build_data(data):
    engineering_transforms = [fe.get_avg_bids,
                              fe.get_unique_devices,
                              fe.get_unique_ips,
                              fe.get_avg_time_between_bids,
                              fe.get_avg_counterbid_times,
                              fe.get_unique_countries]

    features = fe.generate_features(data, engineering_transforms)

    features_and_outcomes = pd.merge(features, data[['bidder_id', 'outcome']], on='bidder_id')
    features_y = features_and_outcomes['outcome']
    del features_and_outcomes['outcome']

    return features_and_outcomes, features_y
