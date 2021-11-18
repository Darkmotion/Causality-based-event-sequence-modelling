import numpy as np
import pandas as pd
import scipy as sc

def get_causality_graph(
        N: int,
        Nmax: int,
        Tmax: int,
        tstep: float,
        dt: float,
        M: float,
        GenerationNum: int,
        D: int,
        kernel = 'gauss',
        w = 2,
        landmark = (0,4,12)
    ) -> nd.array:
    '''
    N: the number of sequences
    Nmax: the maximum number of events per sequence
    Tmax: the maximum size of time window
    dt: the length of each time step
    M: the number of steps in the time interval for computing sup-intensity
    GenerationNum: the number of generations for branch processing
    D: the dimension of Hawkes processes
    kernel: the type of kernels per impact function
    w: the bandwidth of gaussian kernel
    landmark: the central locations of kernels
    '''

    # 1. Approximate simulation of Hawkes processes via branching process
    # 2. Initialize ground truth parameters
    # mu =
    # A =

    # 3. Visualize all impact functions and infectivity matrix

    # 4. Maximum likelihood estimation and basis representation

    # 5. Learning the model by MLE

    # 6. Visualize the infectivity matrix (the adjacent matrix of Granger causality graph)

    return None
