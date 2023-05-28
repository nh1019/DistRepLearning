import networkx as nx
import copy
import torch
import numpy as np

#model aggregation scheme provided by Dr Stefan Vlaski

def generate_graph(n_workers, topology):

  if topology=='star':
    graph = nx.star_graph(n_workers-1)
  elif topology=='random':
    graph = generate_random_graph(n_workers)
  elif topology=='ring':
    graph = generate_ring_graph(n_workers)
  elif topology=='tree':
    graph = nx.random_tree(n=n_workers, seed=0)
  elif topology=='complete':
    graph = nx.complete_graph(n_workers)
  else:
    raise ValueError('Desired topology not implemented.')

  A = nx.adjacency_matrix(graph).todense() + np.eye(n_workers)
  row_sums = np.sum(A, axis=1)
  A = torch.tensor(np.divide(A, row_sums[:, np.newaxis]))
  print(A)
  return A

def generate_random_graph(n_workers):
  connected = 0
  while not connected:
    graph = nx.erdos_renyi_graph(n_workers, p=.6)
    connected = nx.is_connected(graph)

  return graph

def generate_ring_graph(n_workers):
  graph = nx.Graph()

  graph.add_nodes_from(range(n_workers))

  for i in range(n_workers):
    graph.add_edge(i, (i+1)%n_workers)
    graph.add_edge(i, (i-1)%n_workers)

  return graph

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
          print(key)
          print('k: ' + str(k))
          print('l: ' + str(l)) 
          new_weights[k][key] += A[l, k]*weights[l][key]
          print(A[l,k])

  for k in range(n_workers):
    models[k].load_state_dict(new_weights[k])

  return models