# This code defines all the functions in the experiments. 
# Please be sure that you have trained all necessary models and created the CBT plots.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from plotting import plot_scores, plot_circular_graph

def model_comparison(data="simulated", strategy = "Last", t = 2):
    # Compare each model, print p-values.

    # Sigmoid - Double RNN is the ablated version that we benchmark against.
    # We aim to compare and find the best ReMI-Net* variant that outperforms in each case.
    sigmoid_double = np.load(f"./experiments/{data}data_L1_cyclic_sigmoidnorm_double_rnn_model.npy")#[1,0,-1,0]
    minmax_double = np.load(f"./experiments/{data}data_L1_cyclic_minmax_double_rnn_model.npy")#[1,0,-1,0]
    sigmoid_edge = np.load(f"./experiments/{data}data_L1_cyclic_sigmoidnorm_edge_rnn_model.npy")#[1,0,-1,0]
    minmax_edge = np.load(f"./experiments/{data}data_L1_cyclic_minmax_edge_rnn_model.npy")#[1,0,-1,0]

    # CASE 1: BEST MODEL STRATEGY
    if strategy == "Best":
        # P-values between the outperforming method and the benchmark.
        print(ttest_rel(sigmoid_double[1].min(axis=1)[:,t], minmax_edge[1].min(axis=1)[:,t]).pvalue)
        allloss = np.stack([
                            sigmoid_double[1].min(axis=1)[:,t],
                            minmax_double[1].min(axis=1)[:,t],
                            sigmoid_edge[1].min(axis=1)[:,t],
                            minmax_edge[1].min(axis=1)[:,t],
                            ])

    # CASE 2: LAST MODEL STRATEGY
    if strategy == "Last":
        # P-values between the outperforming method and the benchmark.
        print(ttest_rel(minmax_double[1,:,-1,t], minmax_edge[1,:,-1,t]).pvalue)
        allloss = np.stack([
                            sigmoid_double[1,:,-1,t],
                            minmax_double[1,:,-1,t],
                            sigmoid_edge[1,:,-1,t],
                            minmax_edge[1,:,-1,t],
                            ])
    # Plots results.
    plot_scores(allloss,t=t,strategy=strategy,data_type=data)

def track_ad_connectivity_changes(n_folds = 5, hemisphere="LH", norm= "minmax", conv="edge_rnn", label=1):
    # Extract the average connectivity-wise changes into a circular graph.
    # We calculate each difference between the consecutive timepoints and then take the element-wise average across folds.

    diffs = []
    for i in range(n_folds):
        cbt_t0 = np.load(f"./cbt_plots/cbt_{hemisphere}_{norm}_{conv}_time{0}_fold{i}_c{label}.npy")
        cbt_t1 = np.load(f"./cbt_plots/cbt_{hemisphere}_{norm}_{conv}_time{1}_fold{i}_c{label}.npy")
        diff = np.divide((cbt_t1 - cbt_t0), cbt_t0, out=np.zeros_like((cbt_t1 - cbt_t0)), where=cbt_t0!=0) # Handle divide by zero.
        #diff = (cbt_t1 - cbt_t0) / cbt_t0 # Divide by zero represented as infinity in the plot (marked as white).
        diffs.append(diff)
    diffs = np.array(diffs)
    # Average across folds.
    avg_diff = diffs.mean(axis=0)

    # We plot circular graph of connectivity-wise average alteration.
    plot_circular_graph(avg_diff)

    # This will show the adjacency matrix that is used in the circular graph.
    plt.matshow(avg_diff*100) # Show percentage by multiplying with 100.
    plt.colorbar(format= '%.1f%%') # Add percentage sign to the plot.
    plt.show()

    # We return all differences, since they might be subject to another operation.
    return diffs

def track_ad_region_changes(n_folds = 5, hemisphere="LH", norm= "minmax", conv="edge_rnn", label=0):
    # Extract the average region-wise changes.
    # We calculate each difference between the consecutive timepoints and then take the element-wise average across folds and connectivities.

    diffs = []
    for i in range(n_folds):
        cbt_t0 = np.load(f"./cbt_plots/cbt_{hemisphere}_{norm}_{conv}_time{0}_fold{i}_c{label}.npy")
        cbt_t1 = np.load(f"./cbt_plots/cbt_{hemisphere}_{norm}_{conv}_time{1}_fold{i}_c{label}.npy")
        diff = np.divide((cbt_t1 - cbt_t0), cbt_t0, out=np.zeros_like((cbt_t1 - cbt_t0)), where=cbt_t0!=0)
        diffs.append(diff)
    diffs = np.array(diffs)
    # Average across folds.
    avg_diff = diffs.mean(axis=0)

    # We need to take absolute values first in order to avoid negative changes that eliminates the positive ones.
    # Average across connectivities.
    cbt_regions = np.abs(avg_diff).mean(axis=0, keepdims=True)

    TOPK = 5
    ind = np.argpartition(cbt_regions[0], -TOPK)[-TOPK:]
    cbt_selected_features = ind[np.argsort(cbt_regions[0,ind])]
    print(cbt_selected_features)

    # Return aggregated information of region importance for each fold (n_folds, n_regions):(5,35)
    return cbt_regions

if __name__ == "__main__":
    model_comparison()
    # You need real data with hemispheres in order to perform following exeriments:
    # cbt_regions = track_ad_region_changes(hemisphere="LH")
    # print(cbt_regions.mean()*100)
    # cbt_regions = track_ad_region_changes(hemisphere="RH")
    # print(cbt_regions.mean()*100)