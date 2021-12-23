import argparse
import causal_nh.utils as utils
import sys
import os
import datetime
import time
from torch.utils.data import DataLoader
from causal_nh.model.NeuralHawkesCuda import NeuralHawkes, CausalNeuralHawkes, CausalNeuralHawkes_v2
import torch
import matplotlib.pyplot as plt
import test
import numpy as np
import torch.nn.functional as F

def log_valid(path_to_model, time_duration, type_test, seq_lens, device):
    model = torch.load(path_to_model)
    seq_lens = torch.tensor(seq_lens)
    sim_durations, total_time_seqs, time_simulation_index = utils.generate_simulation(time_duration, seq_lens)
    type_test.to(device)
    time_duration.to(device)
    sim_durations.to(device)
    total_time_seqs.to(device)
    seq_lens.to(device)
    time_simulation_index.to(device)
    h_out, c_out, c_bar_out, decay_out, gate_out = model(type_test, time_duration)
    part_one_likelihood, part_two_likelihood, sum_likelihood = model.conttime_loss(h_out, c_out, c_bar_out, decay_out,
                                                                                   gate_out, type_test, sim_durations,
                                                                                   total_time_seqs, seq_lens,
                                                                                   time_simulation_index)
    total_size = torch.sum(seq_lens)
    log_likelihood = torch.sum(part_one_likelihood - part_two_likelihood) / total_size
    type_likelihood = torch.sum(part_one_likelihood - sum_likelihood) / total_size
    time_likelihood = log_likelihood - type_likelihood
    return log_likelihood, type_likelihood, time_likelihood

def type_valid(path_to_model, time_durations, seq_lens_lists, type_tests):
    model = torch.load(path_to_model)
    numb_tests = time_durations.shape[0]
    original_types = []
    predicted_types = []
    for i in range(numb_tests):
        time_duration = time_durations[i:i + 1]
        type_test = type_tests[i:i + 1]
        seq_len = seq_lens_lists[i]

        original_types.append(type_test[0][seq_len].item())
        type_test = type_test[:, :seq_len]
        time_duration = time_duration[:, :seq_len + 1]

        h_out, c_out, c_bar_out, decay_out, gate_out = model(type_test, time_duration)
        lambda_all = F.softplus(model.hidden_lambda(h_out[-1]))
        lambda_sum = torch.sum(lambda_all, dim=-1)
        lambda_all = lambda_all / lambda_sum
        # print(lambda_all)
        _, predict_type = torch.max(lambda_all, dim=-1)
        predicted_types.append(predict_type.item())

    total_numb = len(original_types)
    numb_correct = 0
    for idx in range(total_numb):
        if predicted_types[idx] == original_types[idx]:
            numb_correct += 1
    return numb_correct / total_numb

def type_valid_cnh_v1(path_to_model, time_durations, seq_lens_lists, type_tests, A):
    model = torch.load(path_to_model)
    numb_tests = time_durations.shape[0]
    original_types = []
    predicted_types = []
    for i in range(numb_tests):
        time_duration = time_durations[i:i + 1]
        type_test = type_tests[i:i + 1]
        seq_len = seq_lens_lists[i]

        original_types.append(type_test[0][seq_len].item())
        type_test = type_test[:, :seq_len]
        time_duration = time_duration[:, :seq_len + 1]

        h_out, c_out, c_bar_out, decay_out, gate_out = model(type_test, time_duration)
        lambda_all = F.softplus(torch.matmul(model.hidden_lambda(h_out[-1]),
                                             A.mul(model.weighted_A)))
        lambda_sum = torch.sum(lambda_all, dim=-1)
        lambda_all = lambda_all / lambda_sum
        # print(lambda_all)
        _, predict_type = torch.max(lambda_all, dim=-1)
        predicted_types.append(predict_type.item())

    total_numb = len(original_types)
    numb_correct = 0
    for idx in range(total_numb):
        if predicted_types[idx] == original_types[idx]:
            numb_correct += 1
    return numb_correct / total_numb

def type_valid_cnh_v2(path_to_model, time_durations, seq_lens_lists, type_tests, A):
    model = torch.load(path_to_model)
    numb_tests = time_durations.shape[0]
    original_types = []
    predicted_types = []
    for i in range(numb_tests):
        time_duration = time_durations[i:i + 1]
        type_test = type_tests[i:i + 1]
        seq_len = seq_lens_lists[i]

        original_types.append(type_test[0][seq_len].item())
        type_test = type_test[:, :seq_len]
        time_duration = time_duration[:, :seq_len + 1]

        h_out, c_out, c_bar_out, decay_out, gate_out = model(type_test, time_duration)
        lambda_all = F.softplus(torch.matmul(model.hidden_lambda(h_out[-1]),
                                             A))
        lambda_sum = torch.sum(lambda_all, dim=-1)
        lambda_all = lambda_all / lambda_sum
        # print(lambda_all)
        _, predict_type = torch.max(lambda_all, dim=-1)
        predicted_types.append(predict_type.item())

    total_numb = len(original_types)
    numb_correct = 0
    for idx in range(total_numb):
        if predicted_types[idx] == original_types[idx]:
            numb_correct += 1
    return numb_correct / total_numb

def train_nh(train, dev, path_to_save, used_model, lr=0.1, num_epochs=10, batch_size=10):
    now = str(datetime.datetime.today()).split()
    now = now[0] + "-" + now[1][:5]
    print("Processing data...")

    type_size = train[3]
    train = train[:3]
    dev = dev[:3]

    time_duration, type_train, seq_lens_list = train
    test_duration, type_test, seq_lens_test = dev
    time_duration, type_train = utils.padding_full(time_duration, type_train, seq_lens_list, type_size)
    test_duration, type_test = utils.padding_full(test_duration, type_test, seq_lens_test, type_size)

    train_data = utils.Data_Batch(time_duration, type_train, seq_lens_list)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    print("start training...")
    if used_model:
        model = torch.load("model.pt")
    else:
        model = NeuralHawkes(n_types=type_size, lr=lr)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("You are using GPU acceleration.")
        print("Number of GPU: ", torch.get_num_threads())
    else:
        device = torch.device("cpu")
        print("CUDA is not Available. You are using CPU only.")
        print("Number of cores: ", os.cpu_count())


    loss_value = []

    log_test_list = []
    log_time_list = []
    log_type_list = []

    type_accuracy_list = []

    model.to(device)
    prefix = datetime.datetime.now().strftime('%Y_%m_%H_%M_%S')

    t3 = time.time()

    for i in range(num_epochs):
        loss_total = 0
        events_total = 0
        max_len = len(train_data)
        for idx, a_batch in enumerate(train_data):
            durations, type_items, seq_lens = a_batch['duration_seq'], a_batch['event_seq'], a_batch['seq_len']
            sim_durations, total_time_seqs, time_simulation_index = utils.generate_simulation(durations, seq_lens)
            type_items.to(device)
            durations.to(device)
            sim_durations.to(device)
            total_time_seqs.to(device)
            seq_lens.to(device)
            time_simulation_index.to(device)
            batch = (type_items, durations)
            loss = model.train_batch(batch, sim_durations, total_time_seqs, seq_lens, time_simulation_index)
            log_likelihood = -loss
            total_size = torch.sum(seq_lens)
            loss_total += log_likelihood.item()
            events_total += total_size.item()
            print("Epoch {0}, process {1} out of {2} is done".format(i, idx, max_len))
        avg_log = loss_total / events_total
        loss_value.append(-avg_log)
        print("The log-likelihood at epoch {0}: {1}".format(i, avg_log))
        torch.save(model, path_to_save + f"{prefix}_model.pt")
        path_to_model = path_to_save + f"{prefix}_model.pt"

        print("\nvalidating on log likelihood...")
        log_likelihood, type_likelihood, time_likelihood = log_valid(path_to_model, test_duration, type_test, seq_lens_test, device)
        log_test_list.append(-log_likelihood.item())
        log_type_list.append(-type_likelihood.item())
        log_time_list.append(-time_likelihood.item())

        print("\nvalidating on type prediction accuracy if we know when next event will happen...\n\n")
        accuracy = type_valid(path_to_model, test_duration, seq_lens_test, type_test)
        type_accuracy_list.append(accuracy)

    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
    figure.suptitle("'s Training Figure")
    ax[0].set_xlabel("epochs")
    ax[0].plot(loss_value, label='training loss')
    ax[0].plot(log_test_list, label='testing loss')
    ax[0].legend()
    ax[1].set_xlabel("epochs")
    ax[1].plot(log_type_list, label='testing type loss')
    ax[1].plot(log_time_list, label='testing time loss')
    ax[1].legend()
    ax[2].set_xlabel("epochs")
    ax[2].set_ylabel('accuracy')
    ax[2].set_title('type-validation-accuracy')
    ax[2].plot(type_accuracy_list, label='dev type accuracy')
    plt.subplots_adjust(top=0.85)
    figure.tight_layout()
    plt.savefig(path_to_save+f"{prefix}_training.jpg")

    t4 = time.time()
    training_time = t4 - t3
    print("training done..")
    print("training takes {0} seconds".format(training_time))

    print("Saving training loss and validation data...")
    print("If you have a trained model before this, please combine the previous train_date file to" +
          " generate plots that are able to show the whole training information")
    training_info_file = path_to_save + "training-data-" + prefix + ".txt"
    file = open(training_info_file, 'w')
    file.write("log-likelihood: ")
    file.writelines(str(item) + " " for item in loss_value)
    file.write('\nlog-test-likelihood: ')
    file.writelines(str(item) + " " for item in log_test_list)
    file.write('\nlog-type-likelihood: ')
    file.writelines(str(item) + " " for item in log_type_list)
    file.write('\nlog-time-likelihood: ')
    file.writelines(str(item) + " " for item in log_time_list)
    file.write('\naccuracy: ')
    file.writelines(str(item) + " " for item in type_accuracy_list)
    file.close()

def train_causal_nh(train, dev, A, path_to_save, used_model, lr=0.1, num_epochs=10, batch_size=10):
    now = str(datetime.datetime.today()).split()
    now = now[0] + "-" + now[1][:5]
    print("Processing data...")

    type_size = train[3]
    train = train[:3]
    dev = dev[:3]

    time_duration, type_train, seq_lens_list = train
    test_duration, type_test, seq_lens_test = dev
    time_duration, type_train = utils.padding_full(time_duration, type_train, seq_lens_list, type_size)
    test_duration, type_test = utils.padding_full(test_duration, type_test, seq_lens_test, type_size)


    train_data = utils.Data_Batch(time_duration, type_train, seq_lens_list)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)



    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("You are using GPU acceleration.")
        print("Number of GPU: ", torch.get_num_threads())
    else:
        device = torch.device("cpu")
        print("CUDA is not Available. You are using CPU only.")
        print("Number of cores: ", os.cpu_count())

    A = torch.FloatTensor(A).to('cuda')

    print("start training...")
    if used_model:
        model = torch.load("model.pt")
    else:
        model = CausalNeuralHawkes(A=A, n_types=type_size, lr=lr)

    loss_value = []

    log_test_list = []
    log_time_list = []
    log_type_list = []

    model.to(device)

    prefix = datetime.datetime.now().strftime('%Y_%m_%H_%M_%S')

    t3 = time.time()

    type_accuracy_list = []
    for i in range(num_epochs):
        loss_total = 0
        events_total = 0
        max_len = len(train_data)
        for idx, a_batch in enumerate(train_data):
            durations, type_items, seq_lens = a_batch['duration_seq'], a_batch['event_seq'], a_batch['seq_len']
            sim_durations, total_time_seqs, time_simulation_index = utils.generate_simulation(durations, seq_lens)
            type_items.to(device)
            durations.to(device)
            sim_durations.to(device)
            total_time_seqs.to(device)
            seq_lens.to(device)
            time_simulation_index.to(device)
            batch = (type_items, durations)
            loss = model.train_batch(batch, sim_durations, total_time_seqs, seq_lens, time_simulation_index)
            log_likelihood = -loss
            total_size = torch.sum(seq_lens)
            loss_total += log_likelihood.item()
            events_total += total_size.item()
            # print("Epoch {0}, process {1} out of {2} is done".format(i, idx, max_len))
        avg_log = loss_total / events_total
        loss_value.append(-avg_log)
        print("The log-likelihood at epoch {0}: {1}".format(i, avg_log))
        torch.save(model, path_to_save + f"{prefix}_cousal_v1_model.pt")
        path_to_model = path_to_save + f"{prefix}_cousal_v1_model.pt"

        print("\nvalidating on log likelihood...")
        log_likelihood, type_likelihood, time_likelihood = log_valid(path_to_model, test_duration, type_test, seq_lens_test, device)
        log_test_list.append(-log_likelihood.item())
        log_type_list.append(-type_likelihood.item())
        log_time_list.append(-time_likelihood.item())

        print("\nvalidating on type prediction accuracy if we know when next event will happen...\n\n")
        accuracy = type_valid_cnh_v1(path_to_model, test_duration, seq_lens_test, type_test, A)
        type_accuracy_list.append(accuracy)

    print(model.weighted_A)
    print(A)

    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
    figure.suptitle("'s Training Figure")
    ax[0].set_xlabel("epochs")
    ax[0].plot(loss_value, label='training loss')
    ax[0].plot(log_test_list, label='testing loss')
    ax[0].legend()
    ax[1].set_xlabel("epochs")
    ax[1].plot(log_type_list, label='testing type loss')
    ax[1].plot(log_time_list, label='testing time loss')
    ax[1].legend()
    ax[2].set_xlabel("epochs")
    ax[2].set_ylabel('accuracy')
    ax[2].set_title('type-validation-accuracy')
    ax[2].plot(type_accuracy_list, label='dev type accuracy')
    plt.subplots_adjust(top=0.85)
    figure.tight_layout()
    plt.savefig(path_to_save + f"{prefix}_cousal_v1_training.jpg")
    plt.show()


    print("Saving training loss and validation data...")
    print("If you have a trained model before this, please combine the previous train_date file to" +
          " generate plots that are able to show the whole training information")
    training_info_file = path_to_save + f"training-data-{prefix}_cousal_v1.txt"
    file = open(training_info_file, 'w')
    file.write("log-likelihood: ")
    file.writelines(str(item) + " " for item in loss_value)
    file.write('\nlog-test-likelihood: ')
    file.writelines(str(item) + " " for item in log_test_list)
    file.write('\nlog-type-likelihood: ')
    file.writelines(str(item) + " " for item in log_type_list)
    file.write('\nlog-time-likelihood: ')
    file.writelines(str(item) + " " for item in log_time_list)
    file.write('\naccuracy: ')
    file.writelines(str(item) + " " for item in type_accuracy_list)
    file.close()

def train_causal_nh_v2(train, dev, A, path_to_save, used_model, lr=0.1, num_epochs=10, batch_size=10):
    now = str(datetime.datetime.today()).split()
    now = now[0] + "-" + now[1][:5]
    print("Processing data...")

    type_size = train[3]
    train = train[:3]
    dev = dev[:3]

    time_duration, type_train, seq_lens_list = train
    test_duration, type_test, seq_lens_test = dev
    time_duration, type_train = utils.padding_full(time_duration, type_train, seq_lens_list, type_size)
    test_duration, type_test = utils.padding_full(test_duration, type_test, seq_lens_test, type_size)


    train_data = utils.Data_Batch(time_duration, type_train, seq_lens_list)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)



    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("You are using GPU acceleration.")
        print("Number of GPU: ", torch.get_num_threads())
    else:
        device = torch.device("cpu")
        print("CUDA is not Available. You are using CPU only.")
        print("Number of cores: ", os.cpu_count())


    A = torch.FloatTensor(A).to('cuda')

    print("start training...")
    if used_model:
        model = torch.load("model.pt")
    else:
        model = CausalNeuralHawkes_v2(A=A, n_types=type_size, lr=lr)

    loss_value = []

    log_test_list = []
    log_time_list = []
    log_type_list = []

    model.to(device)

    prefix = datetime.datetime.now().strftime('%Y_%m_%H_%M_%S')

    t3 = time.time()

    type_accuracy_list = []
    for i in range(num_epochs):
        loss_total = 0
        events_total = 0
        max_len = len(train_data)
        for idx, a_batch in enumerate(train_data):
            durations, type_items, seq_lens = a_batch['duration_seq'], a_batch['event_seq'], a_batch['seq_len']
            sim_durations, total_time_seqs, time_simulation_index = utils.generate_simulation(durations, seq_lens)
            type_items.to(device)
            durations.to(device)
            sim_durations.to(device)
            total_time_seqs.to(device)
            seq_lens.to(device)
            time_simulation_index.to(device)
            batch = (type_items, durations)
            loss = model.train_batch(batch, sim_durations, total_time_seqs, seq_lens, time_simulation_index)
            log_likelihood = -loss
            total_size = torch.sum(seq_lens)
            loss_total += log_likelihood.item()
            events_total += total_size.item()
            print("Epoch {0}, process {1} out of {2} is done".format(i, idx, max_len))
        avg_log = loss_total / events_total
        loss_value.append(-avg_log)
        print("The log-likelihood at epoch {0}: {1}".format(i, avg_log))
        torch.save(model, path_to_save + f"{prefix}_solid_A_model.pt")
        path_to_model = path_to_save + f"{prefix}_solid_A_model.pt"

        print("\nvalidating on log likelihood...")
        log_likelihood, type_likelihood, time_likelihood = log_valid(path_to_model, test_duration, type_test, seq_lens_test, device)
        log_test_list.append(-log_likelihood.item())
        log_type_list.append(-type_likelihood.item())
        log_time_list.append(-time_likelihood.item())

        print("\nvalidating on type prediction accuracy if we know when next event will happen...\n\n")
        accuracy = type_valid_cnh_v2(path_to_model, test_duration, seq_lens_test, type_test, A)
        type_accuracy_list.append(accuracy)

    print(A)

    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
    figure.suptitle("'s Training Figure")
    ax[0].set_xlabel("epochs")
    ax[0].plot(loss_value, label='training loss')
    ax[0].plot(log_test_list, label='testing loss')
    ax[0].legend()
    ax[1].set_xlabel("epochs")
    ax[1].plot(log_type_list, label='testing type loss')
    ax[1].plot(log_time_list, label='testing time loss')
    ax[1].legend()
    ax[2].set_xlabel("epochs")
    ax[2].set_ylabel('accuracy')
    ax[2].set_title('type-validation-accuracy')
    ax[2].plot(type_accuracy_list, label='dev type accuracy')
    plt.subplots_adjust(top=0.85)
    figure.tight_layout()
    plt.savefig(path_to_save + f"{prefix}_solid_A_training.jpg")
    plt.show()


    print("Saving training loss and validation data...")
    print("If you have a trained model before this, please combine the previous train_date file to" +
          " generate plots that are able to show the whole training information")
    training_info_file = path_to_save + f"training-data-{prefix}_solid_A_.txt"
    file = open(training_info_file, 'w')
    file.write("log-likelihood: ")
    file.writelines(str(item) + " " for item in loss_value)
    file.write('\nlog-test-likelihood: ')
    file.writelines(str(item) + " " for item in log_test_list)
    file.write('\nlog-type-likelihood: ')
    file.writelines(str(item) + " " for item in log_type_list)
    file.write('\nlog-time-likelihood: ')
    file.writelines(str(item) + " " for item in log_time_list)
    file.write('\naccuracy: ')
    file.writelines(str(item) + " " for item in type_accuracy_list)
    file.close()

