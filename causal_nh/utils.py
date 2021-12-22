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


def padding_full(time_duration, type_train, seq_lens_list, type_size):
    max_len = max(seq_lens_list)
    batch_size = len(time_duration)
    time_duration_padded = torch.zeros(size=(batch_size, max_len+1))
    type_train_padded = torch.zeros(size=(batch_size, max_len+1), dtype=torch.long)
    for idx in range(batch_size):
        time_duration_padded[idx, 1:seq_lens_list[idx]+1] = time_duration[idx]
        type_train_padded[idx, 0] = type_size
        type_train_padded[idx, 1:seq_lens_list[idx]+1] = type_train[idx]
    return time_duration_padded, type_train_padded


class Data_Batch:
    def __init__(self, duration, events, seq_len):
        self.duration = duration
        self.events = events
        self.seq_len = seq_len

    def __len__(self):
        return self.events.shape[0]

    def __getitem__(self, index):
        sample = {
            'event_seq': self.events[index],
            'duration_seq': self.duration[index],
            'seq_len': self.seq_len[index]
        }
        return sample

def generate_simulation(durations, seq_len):
    max_seq_len = max(seq_len)
    simulated_len = max_seq_len * 5
    sim_durations = torch.zeros(durations.shape[0], simulated_len)
    sim_duration_index = torch.zeros(durations.shape[0], simulated_len, dtype=torch.long)
    total_time_seqs = []
    for idx in range(durations.shape[0]):
        time_seq = torch.stack([torch.sum(durations[idx][:i]) for i in range(1, seq_len[idx]+2)])
        total_time = time_seq[-1].item()
        total_time_seqs.append(total_time)
        sim_time_seq, _ = torch.sort(torch.empty(simulated_len).uniform_(0, total_time))
        sim_duration = torch.zeros(simulated_len)

        for idx2 in range(time_seq.shape.__getitem__(-1)):
            duration_index = sim_time_seq > time_seq[idx2].item()
            sim_duration[duration_index] = sim_time_seq[duration_index] - time_seq[idx2]
            sim_duration_index[idx][duration_index] = idx2

        sim_durations[idx, :] = sim_duration[:]
    total_time_seqs = torch.tensor(total_time_seqs)
    return sim_durations, total_time_seqs, sim_duration_index


