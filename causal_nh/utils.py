import pickle
import torch
import numpy as np

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
