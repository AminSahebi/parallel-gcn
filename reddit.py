# coding: utf-8

from networkx.readwrite import json_graph
import networkx as nx
import json
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import os

dataset_dir = '.'
prefix = 'reddit'

def load_data(prefix, normalize=True, load_walks=False):
    with open(prefix + "-G.json") as file:
        G_data = json.load(file)
    G = json_graph.node_link_graph(G_data)
    
    if isinstance(list(G.nodes())[0], int):
        conversion = lambda n: int(n)
    else:
        conversion = lambda n: n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    
    with open(prefix + "-id_map.json") as file:
        id_map = json.load(file)
    id_map = {conversion(k): int(v) for k, v in id_map.items()}
    
    walks = []
    
    with open(prefix + "-class_map.json") as file:
        class_map = json.load(file)
    
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n: n
    else:
        lab_conversion = lambda n: int(n)

    class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}

    broken_count = 0
    for node in list(G.nodes()):
        if not 'val' in G.nodes[node] or not 'test' in G.nodes[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    print("Loaded data.. now preprocessing..")
    for edge in list(G.edges()):
        if (G.nodes[edge[0]]['val'] or G.nodes[edge[1]]['val'] or
            G.nodes[edge[0]]['test'] or G.nodes[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(list(map(conversion, line.split())))

    return G, feats, id_map, walks, class_map

data = load_data(prefix)
(G, feats, id_map, walks, class_map) = data

train_ids = [n for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']]
test_ids = [n for n in G.nodes() if G.nodes[n]['test']]
val_ids = [n for n in G.nodes() if G.nodes[n]['val']]
ids = train_ids + test_ids + val_ids

train_labels = [class_map[i] for i in train_ids]
test_labels = [class_map[i] for i in test_ids]
val_labels = [class_map[i] for i in val_ids]
labels = train_labels + test_labels + val_labels

ids, labels = zip(*sorted(zip(ids, labels)))
name_to_id = {}
for i, name in enumerate(ids):
    name_to_id[name] = i

print(len(train_ids), len(train_labels))
print(len(test_ids), len(test_labels))
print(len(val_ids), len(val_labels))
print(len(ids), len(labels))

graph_file = open(prefix + '.graph', "w")
adj_matrix = {}
for node in G.nodes:
    neighbors = list(G.neighbors(node))
    adj_matrix[name_to_id[node]] = [str(name_to_id[n]) for n in neighbors]

for i in range(len(adj_matrix)):
    print(" ".join(adj_matrix[i]), file=graph_file)
graph_file.close()

split_file = open(prefix + '.split', "w")
split_dict = {}
train_id_set = set(train_ids)
val_id_set = set(val_ids)
test_id_set = set(test_ids)
for i, node in enumerate(G.nodes):
    split = 0
    if node in train_id_set:
        split = 1
    elif node in val_id_set:
        split = 2
    elif node in test_id_set:
        split = 3
    split_dict[name_to_id[node]] = split

for i in range(len(split_dict)):
    split = split_dict[i]
    print(split, file=split_file)

split_file.close()

final_features = []
final_labels = []
for i, current_id in enumerate(ids):
    final_features.append(feats[id_map[current_id]])
    final_labels.append(labels[i])

from sklearn.datasets import dump_svmlight_file
dump_svmlight_file(final_features, final_labels, prefix + ".svmlight")

