import numpy as np
import pandas as pd
import scipy as sc
from tick.hawkes import HawkesADM4, HawkesSumGaussians
from sklearn.preprocessing import MinMaxScaler

def data_preprocessing(data, n_types):
    data_processed = []
    for sequence in data:
        updated_sequence = [[] for _ in range(n_types)]
        for item in sequence:
            updated_sequence[item['type_event']].append(item['time_since_start'])
        updated_sequence = [np.array(item, dtype=np.double) for item in updated_sequence]
        data_processed.append(updated_sequence)
    return data_processed


def get_causality_graph(train, test, type_size, method):
    processed_train = data_preprocessing(train, type_size)
    processed_test = data_preprocessing(test, type_size)

    if method == 'ADM4':
        decay = .3
        estimator_adm4 = HawkesADM4(decay, n_threads=6)
        estimator_adm4.fit(processed_train)
        W = estimator_adm4.adjacency
        # print(method, f'log-likelihood: {estimator_adm4.score(processed_test)}')

    elif method == 'SumGaussians':
        learner = HawkesSumGaussians(5, max_iter=100)
        learner.fit(processed_train)
        W = learner.get_kernel_norms()

    scaler = MinMaxScaler()
    A = scaler.fit_transform(W)
    A[A > 0.5] = 1
    A[A < 0.5] = 0

    return W, A
