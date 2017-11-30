from operator import itemgetter
from itertools import groupby

import pandas as pd
import numpy as np
import matplotlib.transforms as mtransforms

from matplotlib import pyplot as plt


def _move_avg(df, w_size):
    avg_df = df.rolling(w_size).sum()
    return avg_df


def _shift_event(event, dt):
    i = 0
    while i < len(event):
        event[i] = event[i] + dt
        i += 1
    return event


def get_precipitation_events(df, df_original, k=0.3, diff_min=0.1, interval_length=3):
    diff_df = df.diff()
    diff_30_df = diff_df.quantile(axis=1, q=k)
    precip_idx = np.argwhere(diff_30_df > diff_min).flatten()
    l_events = []
    for k, g in groupby(enumerate(precip_idx), lambda (i, x): i - x):
        l = map(itemgetter(1), g)
        if len(l) > 1:
            l_events.append(l)
    i = 0
    while i < len(l_events) - 1:
        l = l_events[i]
        sum_sum = 0
        qk = l[-1] + 1
        while qk < l_events[i + 1][0]:
            sum_diff = sum(diff_df.iloc[qk, :])
            sum_sum += sum_diff
            qk += 1
        if ((l_events[i + 1][0] - l[-1]) <= max(len(l_events[i + 1]), len(l)) and sum_sum >= 0) or (
            l_events[i + 1][0] - l[-1]) <= (len(l_events[i + 1]) + len(l)) / interval_length:
            l_events[i] = l_events[i] + range(l[-1] + 1, l_events[i + 1][0]) + l_events[i + 1]
            del l_events[i + 1]
            if i >= 2:
                i -= 1
        else:
            i += 1
    i = 0
    while i < len(l_events):
        index_i = l_events[i][-1]
        while sum(df_original.iloc[index_i, :]) < sum(df_original.iloc[index_i - 1, :]):
            l_events[i] = _shift_event(l_events[i], -1)
            index_i = l_events[i][-1]
        while sum(df_original.iloc[index_i, :]) < sum(df_original.iloc[index_i + 1, :]):
            l_events[i] = _shift_event(l_events[i], 1)
            index_i = l_events[i][-1]
        i += 1

    return l_events


def get_melting_events(df, k=0.85, diff_min=-0.2, interval_length=5, filter_length=3):
    diff_df = df.diff()
    diff_30_df = diff_df.quantile(axis=1, q=k)
    precip_idx = np.argwhere(diff_30_df < diff_min).flatten()
    l_events = []
    for k, g in groupby(enumerate(precip_idx), lambda (i, x): i - x):
        l = map(itemgetter(1), g)
        if len(l) > 1:
            l_events.append(l)

    i = 0
    while i < len(l_events) - 1:
        l = l_events[i]
        sum_sum = 0
        qk = l[-1] + 1
        l_end = sum(df.iloc[l[-1], :]) / len(df.iloc[l[-1], :])
        l_event_start = sum(df.iloc[l_events[i + 1][0], :]) / len(df.iloc[l_events[i + 1][0], :])
        delta = l_event_start - l_end
        if ((l_events[i + 1][0] - l[-1]) <= interval_length and delta < 0):
            l_events[i] = l_events[i] + range(l[-1] + 1, l_events[i + 1][0]) + l_events[i + 1]
            del l_events[i + 1]
            if i >= 2:
                i -= 1
        else:
            i += 1

    i = 0
    while i <= len(l_events) - 1:
        if len(l_events[i]) <= filter_length:
            del l_events[i]
        else:
            i += 1

    return l_events


def plot_events_ts(df, l_events):
    ax = df.plot(legend=False, figsize=(15, 3))
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for event in l_events:
        ax.fill_between([event[0] - 1, event[-1]], 0, 1, facecolor='green', alpha=0.5, transform=trans)
    plt.show()
