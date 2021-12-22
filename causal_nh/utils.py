import pickle
import torch
import numpy as np
from tick.hawkes import SimuHawkesExpKernels
import pandas as pd
import matplotlib.pyplot as plt

def open_pkl_file(path, description):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
        type_size = data["dim_process"]
        data = data[description]
    time_durations = []
    type_seqs = []
    seq_lens = []
    for i in range(len(data)):
        event_seqs = np.array(
            [
                int(event["type_event"])
                for event in data[i]
                if int(event["type_event"]) <= 1
            ]
        )
        event_seqs = (event_seqs - 1) * (-1)
        type_seqs.append(torch.LongTensor(event_seqs))
        time_durations.append(
            torch.FloatTensor(
                [
                    float(event["time_since_last_event"])
                    for event in data[i]
                    if int(event["type_event"]) <= 1
                ]
            )
        )
        seq_lens.append(
            len(
                torch.LongTensor(
                    [
                        int(event["type_event"])
                        for event in data[i]
                        if int(event["type_event"]) <= 1
                    ]
                )
            )
        )
    return time_durations, type_seqs, seq_lens, type_size


def generate_synthetic_dataset(type_size, adjacency, len_dataset, baseline_intensity=0.5):
    n_nodes = type_size  # dimension of the Hawkes process

    decays = 3 * np.ones((n_nodes, n_nodes))
    baseline = baseline_intensity * np.ones(n_nodes)

    dataset = []
    intensities = []
    for i in range(len_dataset):
        hawkes = SimuHawkesExpKernels(adjacency=adjacency,
                                      decays=decays,
                                      baseline=baseline,
                                      verbose=False)

        run_time = 50
        hawkes.end_time = run_time
        dt = 0.01
        hawkes.track_intensity(dt)
        hawkes.simulate()
        intensities.append(hawkes.tracked_intensity)

        events_df = pd.DataFrame(columns=['type_event', 'time_since_start'])
        for event_type in range(n_nodes):
            for i in hawkes.timestamps[event_type]:
                events_df = events_df.append(pd.DataFrame({'type_event': event_type,
                                                           'time_since_start': i}, index=[0]))

        events_df = events_df.sort_values('time_since_start').reset_index(drop=True)
        events_df['time_since_last_event'] = events_df.time_since_start.diff(1).fillna(
            events_df.time_since_start.values[0])
        events_df['idx_event'] = events_df.index + 1
        dict_to_save = events_df.to_dict(orient='records')
        dataset.append(dict_to_save)

    return dataset, intensities


def plot_sample_ts(dataset, i, dim):
    df = pd.DataFrame([])
    df['idx'] = range(1, len(dataset[i]) + 1)
    df['time_series'] = i
    df['event_type'] = None
    df['time_since_start'] = None
    df['time_since_last_event'] = None
    for entry in dataset[i]:
        df.loc[df['idx'] == entry['idx_event'], 'event_type'] = entry['type_event']
        df.loc[df['idx'] == entry['idx_event'], 'time_since_start'] = entry['time_since_start']
        df.loc[df['idx'] == entry['idx_event'], 'time_since_last_event'] = entry['time_since_last_event']

    plt.figure(figsize=(20, 5), dpi=200)
    plt.title('Sample time series')
    plt.grid(color='gray', linestyle='-', linewidth=0.2)
    for event_type in range(dim):
        times = df.loc[(df.event_type == event_type)].time_since_start
        events = df.loc[(df.event_type == event_type)].event_type
        plt.scatter(times,
                    events,
                    label=f'event_{event_type}')

        plt.vlines(times, [0] * len(events), events, linestyle='--', linewidth=0.5)

    plt.legend(loc='upper right')
    plt.xlabel('Time since start')
    plt.ylabel('Event type')
    plt.show()


