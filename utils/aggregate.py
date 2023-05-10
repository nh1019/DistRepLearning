import networkx as nx
import copy
import torch
import numpy as np

#model aggregation scheme provided by Dr Stefan Vlaski

def generate_graph(n_workers):
  connected = 0
  while not connected:
    graph = nx.barabasi_albert_graph(n_workers, 1)
    connected = nx.is_connected(graph)

  A = nx.adjacency_matrix(graph).todense() + np.eye(n_workers)
  A = torch.tensor(A/np.sum(A, axis=0))
  return A

def aggregate(n_workers, models, A):
  weights = [models[k].state_dict() for k in range(n_workers)]
  
  new_weights = copy.deepcopy(weights)
  for k in range(n_workers):
    for key in new_weights[k].keys():
      if key[len(key)-4:len(key)]=='bias' or key[len(key)-6:len(key)]=='weight':
        new_weights[k][key] *= 0

  for k in range(n_workers):
    for l in range(n_workers):
      for key in new_weights[k].keys():
        if key[len(key)-4:len(key)]=='bias' or key[len(key)-6:len(key)]=='weight':
          new_weights[k][key] += 0.25*weights[l][key]

  for k in range(n_workers):
    models[k].load_state_dict(new_weights[k])

  return models