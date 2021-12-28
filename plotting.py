import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

from matplotlib._color_data import BASE_COLORS
from nxviz.api import CircosPlot

def save_csv(name, data, columns, index):
    saving = pd.DataFrame(data=data,columns=columns,index=index)
    saving.to_csv(name)

def plot_cbt(img, fold_num=1, timepoint=0, dataset="simulated",norm="minmax",conv="edge_rnn"):
    img = np.repeat(np.repeat(img, 10, axis=1), 10, axis=0)
    plt.imshow(img)
    plt.title(f"CBT at Fold {fold_num} - Time {timepoint}")
    plt.axis('off')
    plt.colorbar()
    
    plt.savefig(f"./cbt_plots/{dataset}_{norm}_{conv}_cbt_time{timepoint}_fold{fold_num}.png")
    plt.close()

def plot_training_curve(losses1,losses2):
    plt.plot(np.arange(31)*5, losses1, label="Avg Train Frob Loss")
    plt.plot(np.arange(31)*5, losses2, label="Avg Train Reg Loss")
    plt.legend()
    plt.title("Mean Regularizer Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

def plot_scores(data, t=0, strategy="Last",data_type="simulated"):
    plt.figure()#figsize=(20,10))
    color_list = list(BASE_COLORS.keys())
    color_list.remove("w")
    gap = .8 / len(data)
    labels = []
    for i, row in enumerate(data[0]):
        labels.append("Fold " + str(i+1))
    # Add average column.
    labels.append("Average")
    data = np.concatenate((data, data.mean(axis=1,keepdims=True)), axis=1)
    
    barlabels = ["Cyclic Sigmoid Double RNN","Cyclic Weighted Minmax Double RNN", "Cyclic Sigmoid Edge RNN", "Cyclic Weighted Minmax Edge RNN"]
    ticks = np.arange(data.shape[1])
    for i, row in enumerate(data):
        plt.bar(ticks+i*gap, row, width = gap, edgecolor = "k", color = color_list[i % data.shape[0]], label=barlabels[i])

    plt.xticks(ticks+(data.shape[0]*gap*1/2)-(gap/2), labels)
    plt.title(f"Average Frobenius Loss Time {t+1} - {strategy} Model")
    plt.ylim(top=18.0) #ymax is your value
    plt.ylim(bottom=15) #ymin is your value
    plt.legend()#loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    #plt.show()
    name = f"./experiments/final_{data_type}_{strategy.lower()}model_time{t}"
    save_csv(name+".csv",data.transpose(),barlabels,labels)
    plt.savefig(name + ".png")
    plt.close()

def plot_circular_graph(cbt, n_nodes=35, TOPK=5):
    cbt[np.tril_indices_from(cbt, -1)] = 0
    cbt = np.abs(cbt)
    cbt_selected_features = np.unravel_index(np.argsort(cbt.ravel())[-TOPK:], cbt.shape)
    print(cbt_selected_features)
    node_list=np.arange(n_nodes).tolist()
    edge_list=[]
    for f in range(TOPK):
        i = cbt_selected_features[0][f]
        j = cbt_selected_features[1][f]
        edge_list.append((i,j,cbt[i,j]*100))

    print(edge_list)

    G = nx.Graph()
    G.add_nodes_from(node_list)
    G.add_weighted_edges_from(edge_list)
    color_list=["a", "b", "c", "d", "e"]
    for n, d in G.nodes(data=True):
        G.nodes[n]["class"] = node_list[n-1]

    c = CircosPlot(graph=G,node_labels=True,
        node_label_rotation=True,
        fontsize=15,
        group_legend=False,
        figsize=(7, 7),node_color="class",edge_width='weight')
    c.draw()
    plt.title(f"Right Hemisphere\n", fontdict={'fontsize': 20, 'fontweight': 'medium'})
    plt.show()