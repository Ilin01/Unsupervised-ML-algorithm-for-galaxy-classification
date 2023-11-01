import pandas as pd
import numpy as np
from neupy.algorithms import GrowingNeuralGas
from sklearn.preprocessing import MinMaxScaler


def gng_func(data_init, max_nodes,n_iter_before_neuron_added=100,subsample_size=100000,run_number=0):
    gng=GrowingNeuralGas(
        n_inputs=len(data_init.values[0]),
        n_start_nodes=2,

        shuffle_data=True,
        verbose=True,

        step=0.2,
        neighbour_step=0.006,
        show_epoch=1, 

        max_edge_age=50,
        max_nodes=max_nodes,

        n_iter_before_neuron_added=n_iter_before_neuron_added,
        after_split_error_decay_rate=0.5,
        error_decay_rate=0.995,
        min_distance_for_update=0,
    )

    epochs=10
    data=data_init
    for epoch in range(epochs):
        df_data1 = data[:subsample_size]
        data1=df_data1.values
        indices=df_data1.index
        gng.train(data1, epochs=1)
        data=data.drop(indices)
        print('model '+str(run_number)+': number of nodes so far: ',len(gng.graph.nodes))
        if len(data)<subsample_size:
            break
    
    weights1=[]
    for node_1 in gng.graph.nodes:
        weights1.append(node_1.weight[0])
    weights=np.array(weights1)



    return weights


def gng_model(run_number,samples):

    print('model '+str(run_number)+': CREATING THE GNG MODEL')

    

    samples[samples==0.]=10**(-10)

    samples=np.log(samples)

    scaler=MinMaxScaler()
    scaled_samples=scaler.fit_transform(samples)



    scaled = pd.DataFrame(scaled_samples)

    # maximum number of nodes for the GNG code to generate
    max_nodes=100000

    # subsample size for each iteration
    subsample_size=200000

    nodes=gng_func(scaled,max_nodes=max_nodes,subsample_size=subsample_size,run_number=run_number)

    print('model '+str(run_number)+': THE GNG MODEL HAS BEEN CREATED')


    return nodes,scaled_samples

