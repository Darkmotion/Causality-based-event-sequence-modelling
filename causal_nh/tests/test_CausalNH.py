import torch
from causal_nh.model.NeuralHawkes import CausalNeuralHawkesTrainableWeighted

train_type = torch.tensor([[4, 0, 3, 2, 1, 0, 3, 2, 1], [4, 0, 3, 2, 1, 0, 3, 0, 0]])
train_dtime = torch.tensor(
    [
        [0, 0.5, 0.4, 0.3, 0.2, 0.5, 0.4, 0.3, 0.2],
        [0, 0.5, 0.4, 0.3, 0.2, 0.5, 0.4, 0, 0],
    ]
)
sim_durations = torch.tensor(
    [
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ]
)
sim_index = torch.tensor(
    [[0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 5, 7], [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 7]]
)

A = torch.tensor([[1, 0, 0, 1],
                  [0, 1, 1, 0],
                  [1, 0, 1, 0],
                  [0, 1, 0, 1]])
seq_len_list = torch.tensor([8, 6])
total_time_list = torch.tensor([2.8, 2.8])
# input to initialize the model include number of types
model = CausalNeuralHawkesTrainableWeighted(4, A=A, W=A*0.2)
print('CausalNeuralHawkesTrainableWeighted model initialized with the following structure:')
print(model)
a_batch = (train_type, train_dtime)
# The input to train include batch(padded time, padded type), padded_sim_durations, total_time_list, seq_len_list
model.train_batch(a_batch, sim_durations, total_time_list, seq_len_list, sim_index)
print('Successfully passed the test')
