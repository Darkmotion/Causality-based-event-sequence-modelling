import argparse
from causal_nh.utils import *
import sys
import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from causal_nh.model.NeuralHawkes import NeuralHawkes
from sklearn.metrics import accuracy_score

def simulate_prediction_original_nh(time_durations,
                                    type_tests,
                                    n_samples,
                                    model):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("You are using GPU acceleration.")
        print("Number of GPU: ", torch.get_num_threads())
    else:
        device = torch.device("cpu")
        print("CUDA is not Available. You are using CPU only.")
        print("Number of cores: ", os.cpu_count())

    model.to(device)
    max_duration = torch.max(time_durations).to(device)

    simulated_duration = torch.sort(torch.empty(n_samples).uniform_(0, 40 * max_duration.item()))[0].reshape(
        n_samples, 1)
    time_durations = time_durations.expand(n_samples, time_durations.shape[-1])
    time_duration_sim_padded = torch.cat((time_durations, simulated_duration), dim=1)
    type_tests = type_tests.expand(n_samples, type_tests.shape[-1])

    # get the simulated out and processing h, c, c_bar, decay, o
    h_out, c_out, c_bar_out, decay_out, gate_out = model(type_tests, time_duration_sim_padded)

    simulated_h, simulated_c, simulated_c_bar, simulated_decay, simulated_o = h_out[-1], c_out[-1], c_bar_out[-1], \
                                                                              decay_out[-1], gate_out[-1]
    h_last, c_last, c_bar_last, decay_last, o_last = h_out[-2][0], c_out[-2][0], c_bar_out[-2][0], decay_out[-2][0], \
                                                     gate_out[-2][0]
    #########################

    if isinstance(model, NeuralHawkes):
        estimated_lambda_sum = torch.sum(F.softplus(model.hidden_lambda(simulated_h)), dim=-1)
    else:
        estimated_lambda_sum = torch.sum(F.softplus(torch.matmul(model.hidden_lambda(simulated_h), model.weighted_A.t())), dim=-1)
    estimated_lambda_sum = estimated_lambda_sum.reshape(n_samples)
    simulated_duration = simulated_duration.reshape(n_samples).to(device)
    simulated_integral_exp_terms = torch.stack(
        [(torch.sum(estimated_lambda_sum[:(i + 1)]) * (simulated_duration[i] / (i + 1))) for i in
         range(0, n_samples)])
    simulated_density = estimated_lambda_sum * torch.exp(-simulated_integral_exp_terms)
    simulated_density.to(device)
    estimated_time = torch.sum(simulated_duration * simulated_density) * (40 * max_duration.item()) / n_samples

    # calculate intensity and types
    type_tests = type_tests.to(device)
    type_input = model.emb(type_tests[0][-1])
    cell_i, cell_bar_updated, gate_decay, gate_output = model.lstm_cell(type_input, h_last, c_last, c_bar_last)
    _, hidden = model.lstm_cell.decay(cell_i, cell_bar_updated, gate_decay, gate_output, estimated_time)

    #########################
    if isinstance(model, NeuralHawkes):
        lambda_all = F.softplus(model.hidden_lambda(hidden))
    else:
        lambda_all = F.softplus(torch.matmul(model.hidden_lambda(hidden), model.weighted_A.t()))
    _, estimated_type = torch.max(lambda_all, dim=-1)
    lambda_sum = torch.sum(lambda_all, dim=-1)
    return estimated_time, estimated_type, lambda_sum


def test_prediction(test_data, idx_test_series, n_samples, start_test_idx, path_to_save, path_to_model, dataset_name):
    # read the model and simulate durations for integral
    dim_process = test_data[3]
    test_data = test_data[:3]

    model_name = path_to_model.split('/')[-1].split('.')[0]
    print(model_name)

    time_durations, type_tests, seq_lens_lists = test_data
    time_duration, type_test = padding_full(time_durations, type_tests, seq_lens_lists, dim_process)

    time_duration = time_duration[idx_test_series]
    type_test = type_test[idx_test_series]

    max_len = seq_lens_lists[idx_test_series]
    estimated_times = []
    estimated_intensities = []
    estimated_types = []
    original_time = time_duration[start_test_idx:max_len].tolist()
    model = torch.load(path_to_model)
    for idx in range(start_test_idx, max_len):
        time_durations = time_duration[:idx]
        type_tests = type_test[:idx]

        estimated_time, estimated_type, lambda_sum = simulate_prediction_original_nh(time_durations,
                                                                                     type_tests,
                                                                                     n_samples,
                                                                                     model)

        estimated_times.append(estimated_time.item())
        estimated_types.append(estimated_type.item())
        estimated_intensities.append(lambda_sum.item())
        print("prediction at event {0} on a sequence of length {1} is done".format(idx, max_len))

    rmse = sqrt(mean_squared_error(original_time, estimated_times))

    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
    figure.suptitle(dataset_name + " by Neural Hawkes")
    ax[0].plot(original_time, label="actual")
    ax[0].plot(estimated_times, label="predicted")
    ax[0].set_xlabel("Time Index")
    ax[0].set_ylabel("Time Duration")
    ax[0].legend()
    ax[1].plot(estimated_intensities)
    ax[1].set_xlabel("Time Index")
    ax[1].set_ylabel("Intensity")
    ax[2].bar(x=1, height=rmse)
    ax[2].set_title("time RMSE")
    ax[2].annotate(str(round(rmse, 3)), xy=[1, rmse])
    plt.subplots_adjust(top=0.45)
    figure.tight_layout()
    plt.savefig(path_to_save + f"{model_name}_result.png")
    plt.show()
    original_type = type_test[start_test_idx:max_len]

    df = pd.DataFrame(columns=['time', 'events'])
    df['time'] = original_time
    df['events'] = original_type

    df['time_est'] = estimated_times
    df['events_est'] = estimated_types

    df['time_since_start'] = df['time'].cumsum()
    df['time_since_start_est'] = np.append(
        df['time_est'][0:1].values, df['time_est'][1:].values + df['time_since_start'][:-1].values)
    print(df.events)

    plt.figure(figsize=(20, 5), dpi=200)
    plt.title(f'Comparison of predictions for time series {idx_test_series} from {dataset_name} dataset')
    plt.grid(color='gray', linestyle='-', linewidth=0.2)
    events = np.array(original_type)
    time_since_start = df['time_since_start'].values
    for event_type in range(dim_process):
        idxs = np.where(events == event_type)[0]
        plt.scatter(time_since_start[idxs], events[idxs], label=f'event_{event_type}_original', c='b')
        plt.vlines(time_since_start[idxs], [0] * len(events[idxs]), events[idxs], linestyle='--', linewidth=0.5)
    events = np.array(estimated_types)
    time_since_start = df['time_since_start_est'].values
    for event_type in range(dim_process):
        idxs = np.where(events == event_type)[0]
        plt.scatter(time_since_start[idxs], events[idxs], label=f'event_{event_type}_estimated', c='r')
        plt.vlines(time_since_start[idxs], [0] * len(events[idxs]), events[idxs], linestyle='--', linewidth=0.5)

    plt.legend()
    plt.xlabel('Time durations')
    plt.ylabel('Event occurance')
    plt.savefig(path_to_save + f"{model_name}_result_prediction.png")
    plt.show()
    acc = accuracy_score(original_type, estimated_types)

    return original_time, original_type, estimated_times, estimated_types, estimated_intensities, rmse, acc
