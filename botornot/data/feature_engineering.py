import pandas as pd
import numpy as np


def generate_features(data, transforms):
    agg_data = data['bidder_id']

    for index, transform in enumerate(transforms):
        print("Generating feature ", index+1, " of ", len(transforms))
        agg_data = pd.merge(agg_data, transform(data), on='bidder_id')

    return agg_data


def get_total_bids(joined_data):
    grouped_bidder = joined_data.groupby('bidder_id')
    total_bids = grouped_bidder.size().reset_index(name='total_bids')
    return total_bids


def get_avg_bids(joined_data):
    grouped_bidder_auction = joined_data.groupby(['bidder_id', 'auction'])
    avg_bids = grouped_bidder_auction.size().reset_index(name='bids').groupby('bidder_id').mean().reset_index().rename(
        columns={"bids": 'avg_bids'})
    return avg_bids


def get_unique_devices(joined_data):
    grouped_bidder = joined_data.groupby('bidder_id')
    unique_devices = grouped_bidder.device.nunique().reset_index(name='unique_devices')
    return unique_devices


def get_unique_countries(joined_data):
    grouped_bidder = joined_data.groupby('bidder_id')
    unique_countries = grouped_bidder.country.nunique().reset_index(name='unique_countries')
    return unique_countries


def get_unique_ips(joined_data):
    grouped_bidder = joined_data.groupby('bidder_id')
    unique_ips = grouped_bidder.ip.nunique().reset_index(name='unique_ips')
    return unique_ips


def get_avg_time_between_bids(joined_data):
    grouped_bidder = joined_data.groupby('bidder_id')
    avg_time_between_bids = grouped_bidder.apply(average_time_between_bids).reset_index(name='avg_time_between_bids')
    return avg_time_between_bids


def average_time_between_bids(data):
    times = np.sort(data['time'].tolist())
    diffs = np.diff(times)
    return -1 if len(diffs) == 0 else np.mean(diffs)


def get_avg_counterbid_times(joined_data):
    grouped_auctions = joined_data.groupby('auction')
    counterbid_times = grouped_auctions.apply(counterbid_time)
    real_counterbids = counterbid_times[counterbid_times['counterbid_time'] > -1]
    avg_counterbid_times = real_counterbids.groupby('bidder_id').mean().reset_index()
    return avg_counterbid_times


def counterbid_time(data):
    sorted_by_time = data[['bidder_id', 'time']].sort_values(by='time')
    diffs = np.diff(sorted_by_time['time'].tolist())
    offset = np.insert(diffs, 0, -1)
    data['counterbid_time'] = offset
    return data[['bidder_id', 'counterbid_time']]
