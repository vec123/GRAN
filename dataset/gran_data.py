import torch
import time
import os
import pickle
import glob
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
from utils.data_helper import *


class GRANData(object):

  def __init__(self, config, graphs, tag='train'):
    self.config = config
    self.data_path = config.dataset.data_path
    self.model_name = config.model.name
    self.max_num_nodes = config.model.max_num_nodes
    self.block_size = config.model.block_size
    self.stride = config.model.sample_stride

    self.graphs = graphs
    self.num_graphs = len(graphs)
    self.npr = np.random.RandomState(config.seed)
    self.node_order = config.dataset.node_order
    self.num_canonical_order = config.model.num_canonical_order
    self.tag = tag
    self.num_fwd_pass = config.dataset.num_fwd_pass
    self.is_sample_subgraph = config.dataset.is_sample_subgraph
    self.num_subgraph_batch = config.dataset.num_subgraph_batch
    self.is_overwrite_precompute = config.dataset.is_overwrite_precompute

    if self.is_sample_subgraph:
      assert self.num_subgraph_batch > 0

    self.save_path = os.path.join(
        self.data_path, '{}_{}_{}_{}_{}_{}_{}_precompute'.format(
            config.model.name, config.dataset.name, tag, self.block_size,
            self.stride, self.num_canonical_order, self.node_order))

    if not os.path.isdir(self.save_path) or self.is_overwrite_precompute:
      self.file_names = []
      if not os.path.isdir(self.save_path):
        os.makedirs(self.save_path)

      self.config.dataset.save_path = self.save_path
      for index in tqdm(range(self.num_graphs)):
        G = self.graphs[index]
        data = self._get_graph_data(G)
        tmp_path = os.path.join(self.save_path, '{}_{}.p'.format(tag, index))
        pickle.dump(data, open(tmp_path, 'wb'))
        self.file_names += [tmp_path]
    else:
      self.file_names = glob.glob(os.path.join(self.save_path, '*.p'))

  def _get_graph_data(self, G):
    node_degree_list = [(n, d) for n, d in G.degree()]

    adj_0 = np.array(nx.to_numpy_matrix(G))

    ### Degree descent ranking
    # N.B.: largest-degree node may not be unique
    degree_sequence = sorted(
        node_degree_list, key=lambda tt: tt[1], reverse=True)
    adj_1 = np.array(
        nx.to_numpy_matrix(G, nodelist=[dd[0] for dd in degree_sequence]))

    ### Degree ascent ranking
    degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1])
    adj_2 = np.array(
        nx.to_numpy_matrix(G, nodelist=[dd[0] for dd in degree_sequence]))

    ### BFS & DFS from largest-degree node
    CGs = [G.subgraph(c) for c in nx.connected_components(G)]

    # rank connected componets from large to small size
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

    node_list_bfs = []
    node_list_dfs = []
    for ii in range(len(CGs)):
      node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
      degree_sequence = sorted(
          node_degree_list, key=lambda tt: tt[1], reverse=True)

      bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
      dfs_tree = nx.dfs_tree(CGs[ii], source=degree_sequence[0][0])

      node_list_bfs += list(bfs_tree.nodes())
      node_list_dfs += list(dfs_tree.nodes())

    adj_3 = np.array(nx.to_numpy_matrix(G, nodelist=node_list_bfs))
    adj_4 = np.array(nx.to_numpy_matrix(G, nodelist=node_list_dfs))

    ### k-core
    num_core = nx.core_number(G)
    core_order_list = sorted(list(set(num_core.values())), reverse=True)
    degree_dict = dict(G.degree())
    core_to_node = defaultdict(list)
    for nn, kk in num_core.items():
      core_to_node[kk] += [nn]

    node_list = []
    for kk in core_order_list:
      sort_node_tuple = sorted(
          [(nn, degree_dict[nn]) for nn in core_to_node[kk]],
          key=lambda tt: tt[1],
          reverse=True)
      node_list += [nn for nn, dd in sort_node_tuple]

    adj_5 = np.array(nx.to_numpy_matrix(G, nodelist=node_list))

    if self.num_canonical_order == 5:
      adj_list = [adj_0, adj_1, adj_3, adj_4, adj_5]
    else:
      if self.node_order == 'degree_decent':
        adj_list = [adj_1]
      elif self.node_order == 'degree_accent':
        adj_list = [adj_2]
      elif self.node_order == 'BFS':
        adj_list = [adj_3]
      elif self.node_order == 'DFS':
        adj_list = [adj_4]
      elif self.node_order == 'k_core':
        adj_list = [adj_5]
      elif self.node_order == 'DFS+BFS':
        adj_list = [adj_4, adj_3]
      elif self.node_order == 'DFS+BFS+k_core':
        adj_list = [adj_4, adj_3, adj_5]
      elif self.node_order == 'DFS+BFS+k_core+degree_decent':
        adj_list = [adj_4, adj_3, adj_5, adj_1]
      elif self.node_order == 'all':
        adj_list = [adj_4, adj_3, adj_5, adj_1, adj_0]
      else:
        adj_list = [adj_0]

    # print('number of nodes = {}'.format(adj_0.shape[0]))

    return adj_list

  def __getitem__(self, index):
    K = self.block_size
    N = self.max_num_nodes
    S = self.stride

    # load graph
    print("loading graph")
    print(index)
    print("file_names")
    print(self.file_names[index])
    adj_list = pickle.load(open(self.file_names[index], 'rb'))
    num_nodes = adj_list[0].shape[0]
    num_subgraphs = int(np.floor((num_nodes - K) / S) + 1)

    if self.is_sample_subgraph:
      if self.num_subgraph_batch < num_subgraphs:
        num_subgraphs_pass = int(
            np.floor(self.num_subgraph_batch / self.num_fwd_pass))
      else:
        num_subgraphs_pass = int(np.floor(num_subgraphs / self.num_fwd_pass))

      end_idx = min(num_subgraphs, self.num_subgraph_batch)
    else:
      num_subgraphs_pass = int(np.floor(num_subgraphs / self.num_fwd_pass))
      end_idx = num_subgraphs

    ### random permute subgraph
    rand_perm_idx = self.npr.permutation(num_subgraphs).tolist()

    start_time = time.time()
    data_batch = []
    for ff in range(self.num_fwd_pass):
      ff_idx_start = num_subgraphs_pass * ff
      if ff == self.num_fwd_pass - 1:
        ff_idx_end = end_idx
      else:
        ff_idx_end = (ff + 1) * num_subgraphs_pass

      rand_idx = rand_perm_idx[ff_idx_start:ff_idx_end]

      edges = []
      node_idx_gnn = []
      node_idx_feat = []
      label = []
      subgraph_size = []
      subgraph_idx = []
      att_idx = []
      subgraph_count = 0

      for ii in range(len(adj_list)):
        # loop over different orderings
        adj_full = adj_list[ii]
        # adj_tril = np.tril(adj_full, k=-1)
        print("full adjacency")
        print(adj_full)
        idx = -1
        for jj in range(0, num_nodes, S):
          # loop over different subgraphs
          idx += 1

          ### for each size-(jj+K) subgraph, we generate edges for the new block of K nodes
          if jj + K > num_nodes:
            break

          if idx not in rand_idx:
            continue

          ### get graph for GNN propagation
          adj_block = np.pad(
              adj_full[:jj, :jj], ((0, K), (0, K)),
              'constant',
              constant_values=1.0)  # assuming fully connected for the new block
          adj_block = np.tril(adj_block, k=-1)
          adj_block = adj_block + adj_block.transpose()
          adj_block = torch.from_numpy(adj_block).to_sparse()

          print("subgraph adj_block: ", adj_block)
          dense_adj_block = torch.tensor(adj_block).to_dense()
          print("subgraph dense_adj_block: ", dense_adj_block)

          edges += [adj_block.coalesce().indices().long()]
          
          print("subgraph edges: ", edges)

          ### get attention index
          # exist node: 0
          # newly added node: 1, ..., K
          if jj == 0:
            att_idx += [np.arange(1, K + 1).astype(np.uint8)]
          else:
            att_idx += [
                np.concatenate([
                    np.zeros(jj).astype(np.uint8),
                    np.arange(1, K + 1).astype(np.uint8)
                ])
            ]

          ### get node feature index for GNN input
          # use inf to indicate the newly added nodes where input feature is zero
          if jj == 0:
            node_idx_feat += [np.ones(K) * np.inf]
          else:
            node_idx_feat += [
                np.concatenate([np.arange(jj) + ii * N,
                                np.ones(K) * np.inf])
            ]
          print("subgraph node_idx_feat", node_idx_feat)

          ### get node index for GNN output
          idx_row_gnn, idx_col_gnn = np.meshgrid(
              np.arange(jj, jj + K), np.arange(jj + K))
          idx_row_gnn = idx_row_gnn.reshape(-1, 1)
          idx_col_gnn = idx_col_gnn.reshape(-1, 1)
          node_idx_gnn += [
              np.concatenate([idx_row_gnn, idx_col_gnn],
                             axis=1).astype(np.int64)
          ]
          print("subgraph node_idx_gnn: ", node_idx_gnn)


          ### get predict label
          label += [
              adj_full[idx_row_gnn, idx_col_gnn].flatten().astype(np.uint8)
          ]
          print("subgraph label: ", label)

          subgraph_size += [jj + K]
          subgraph_idx += [
              np.ones_like(label[-1]).astype(np.int64) * subgraph_count
          ]
          subgraph_count += 1

      ### adjust index basis for the selected subgraphs
      cum_size = np.cumsum([0] + subgraph_size).astype(np.int64)
      for ii in range(len(edges)):
        edges[ii] = edges[ii] + cum_size[ii]
        node_idx_gnn[ii] = node_idx_gnn[ii] + cum_size[ii]

      ### pack tensors
      data = {}
      data['adj'] = np.tril(np.stack(adj_list, axis=0), k=-1)
      data['edges'] = torch.cat(edges, dim=1).t().long()
      data['node_idx_gnn'] = np.concatenate(node_idx_gnn)
      data['node_idx_feat'] = np.concatenate(node_idx_feat)
      data['label'] = np.concatenate(label)
      data['att_idx'] = np.concatenate(att_idx)
      data['subgraph_idx'] = np.concatenate(subgraph_idx)
      data['subgraph_count'] = subgraph_count
      data['num_nodes'] = num_nodes
      data['subgraph_size'] = subgraph_size
      data['num_count'] = sum(subgraph_size)
      data_batch += [data]

    end_time = time.time()

    return data_batch

  def __len__(self):
    return self.num_graphs

  def collate_fn(self, batch):
    assert isinstance(batch, list)
    start_time = time.time()
    batch_size = len(batch)
    N = self.max_num_nodes
    C = self.num_canonical_order
    label_list = [] #add label list
    batch_data = []

    for ff in range(self.num_fwd_pass):
      data = {}
      batch_pass = []
      for bb in batch:
        batch_pass += [bb[ff]]

      pad_size = [self.max_num_nodes - bb['num_nodes'] for bb in batch_pass]
      subgraph_idx_base = np.array([0] +
                                   [bb['subgraph_count'] for bb in batch_pass])
      subgraph_idx_base = np.cumsum(subgraph_idx_base)

      data['subgraph_idx_base'] = torch.from_numpy(
        subgraph_idx_base)

      data['num_nodes_gt'] = torch.from_numpy(
          np.array([bb['num_nodes'] for bb in batch_pass])).long().view(-1)

      data['adj'] = torch.from_numpy(
          np.stack(
              [
                  np.pad(
                      bb['adj'], ((0, 0), (0, pad_size[ii]), (0, pad_size[ii])),
                      'constant',
                      constant_values=0.0) for ii, bb in enumerate(batch_pass)
              ],
              axis=0)).float()  # B X C X N X N
      
      #add label to data
      for bb in batch_pass:
            label_list.append(bb['label'])
      data['label'] = torch.from_numpy(np.concatenate(label_list)).float()

      idx_base = np.array([0] + [bb['num_count'] for bb in batch_pass])
      idx_base = np.cumsum(idx_base)

      data['edges'] = torch.cat(
          [bb['edges'] + idx_base[ii] for ii, bb in enumerate(batch_pass)],
          dim=0).long()

      data['node_idx_gnn'] = torch.from_numpy(
          np.concatenate(
              [
                  bb['node_idx_gnn'] + idx_base[ii]
                  for ii, bb in enumerate(batch_pass)
              ],
              axis=0)).long()

      data['att_idx'] = torch.from_numpy(
          np.concatenate([bb['att_idx'] for bb in batch_pass], axis=0)).long()

      # shift one position for padding 0-th row feature in the model
      node_idx_feat = np.concatenate(
          [
              bb['node_idx_feat'] + ii * C * N
              for ii, bb in enumerate(batch_pass)
          ],
          axis=0) + 1
      node_idx_feat[np.isinf(node_idx_feat)] = 0
      node_idx_feat = node_idx_feat.astype(np.int64)
      data['node_idx_feat'] = torch.from_numpy(node_idx_feat).long()

      data['label'] = torch.from_numpy(
          np.concatenate([bb['label'] for bb in batch_pass])).float()

      data['subgraph_idx'] = torch.from_numpy(
          np.concatenate([
              bb['subgraph_idx'] + subgraph_idx_base[ii]
              for ii, bb in enumerate(batch_pass)
          ])).long()

      batch_data += [data]

    end_time = time.time()
    # print('collate time = {}'.format(end_time - start_time))

    return batch_data


class GRANData_node_attributes(object):

  def __init__(self, config, graphs, tag='train'):
    self.config = config
    self.data_path = config.dataset.data_path
    self.model_name = config.model.name
    self.max_num_nodes = config.model.max_num_nodes
    self.block_size = config.model.block_size
    self.stride = config.model.sample_stride

    self.graphs = graphs
    self.num_graphs = len(graphs)
    self.npr = np.random.RandomState(config.seed)
    self.node_order = config.dataset.node_order
    self.num_canonical_order = config.model.num_canonical_order
    self.tag = tag
    self.num_fwd_pass = config.dataset.num_fwd_pass
    self.is_sample_subgraph = config.dataset.is_sample_subgraph
    self.num_subgraph_batch = config.dataset.num_subgraph_batch
    self.is_overwrite_precompute = config.dataset.is_overwrite_precompute

    if self.is_sample_subgraph:
      assert self.num_subgraph_batch > 0

    self.save_path = os.path.join(
        self.data_path, '{}_{}_{}_{}_{}_{}_{}_precompute'.format(
            config.model.name, config.dataset.name, tag, self.block_size,
            self.stride, self.num_canonical_order, self.node_order))

    if not os.path.isdir(self.save_path) or self.is_overwrite_precompute:
      self.file_names_adj = []
      self.file_names_att = []
      if not os.path.isdir(self.save_path):
        os.makedirs(self.save_path)

      self.config.dataset.save_path = self.save_path
      for index in tqdm(range(self.num_graphs)):
        G = self.graphs[index]
        adj_data, att_data = self._get_graph_data(G)        
        tmp_path_att = os.path.join(self.save_path, 'att_{}_{}.p'.format(tag, index))
        pickle.dump(att_data, open(tmp_path_att, 'wb'))
        tmp_path_adj = os.path.join(self.save_path, 'adj_{}_{}.p'.format(tag, index))
        pickle.dump(adj_data, open(tmp_path_adj, 'wb'))
        self.file_names_adj += [tmp_path_adj]
        self.file_names_att += [tmp_path_att]
    else:
      self.file_names_adj = glob.glob(os.path.join(self.save_path, 'adj_*.p'))
      self.file_names_att = glob.glob(os.path.join(self.save_path, 'att_*.p'))

  def _get_graph_data(self, G):
    
    node_degree_list = [(n, d) for n, d in G.degree()]

    adj_0 = np.array(nx.to_numpy_matrix(G))
    
    att_0 = self.get_attributes(G, order=0)

    ### Degree descent ranking
    # N.B.: largest-degree node may not be unique
    degree_sequence = sorted(
        node_degree_list, key=lambda tt: tt[1], reverse=True)
    adj_1 = np.array(
        nx.to_numpy_matrix(G, nodelist=[dd[0] for dd in degree_sequence]))
    
    node_order = [t[0] for t in degree_sequence]
    att_1 = self.get_attributes(G, order=node_order)


    ### Degree ascent ranking
    degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1])
    adj_2 = np.array(
        nx.to_numpy_matrix(G, nodelist=[dd[0] for dd in degree_sequence]))
    node_order = [t[0] for t in degree_sequence]
    att_2 = self.get_attributes(G, order=node_order)

    ### BFS & DFS from largest-degree node
    CGs = [G.subgraph(c) for c in nx.connected_components(G)]

    # rank connected componets from large to small size
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

    node_list_bfs = []
    node_list_dfs = []

    for ii in range(len(CGs)):
      node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
     
      degree_sequence = sorted(
          node_degree_list, key=lambda tt: tt[1], reverse=True)
      
      bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
      dfs_tree = nx.dfs_tree(CGs[ii], source=degree_sequence[0][0])

      node_list_bfs += list(bfs_tree.nodes())
      node_list_dfs += list(dfs_tree.nodes())

    adj_3 = np.array(nx.to_numpy_matrix(G, nodelist=node_list_bfs))
    adj_4 = np.array(nx.to_numpy_matrix(G, nodelist=node_list_dfs))

    att_3 = self.get_attributes(CGs[ii], order=node_list_bfs)
    att_4 = self.get_attributes(CGs[ii], order=node_list_dfs)

    ### k-core
    num_core = nx.core_number(G)
    core_order_list = sorted(list(set(num_core.values())), reverse=True)
    degree_dict = dict(G.degree())
    core_to_node = defaultdict(list)
    for nn, kk in num_core.items():
      core_to_node[kk] += [nn]

    node_list = []
    for kk in core_order_list:
      sort_node_tuple = sorted(
          [(nn, degree_dict[nn]) for nn in core_to_node[kk]],
          key=lambda tt: tt[1],
          reverse=True)
      node_list += [nn for nn, dd in sort_node_tuple]

    adj_5 = np.array(nx.to_numpy_matrix(G, nodelist=node_list))
    att_5 = self.get_attributes(G, order=node_list)

    if self.num_canonical_order == 5:
      adj_list = [adj_0, adj_1, adj_3, adj_4, adj_5]
      att_list = [att_0, att_1, att_3, att_4, att_5]
    else:
      if self.node_order == 'degree_decent':
        adj_list = [adj_1]
        att_list = [att_1]
      elif self.node_order == 'degree_accent':
        adj_list = [adj_2]
        att_list = [att_2]
      elif self.node_order == 'BFS':
        adj_list = [adj_3]
        att_list = [att_3]
      elif self.node_order == 'DFS':
        adj_list = [adj_4]
        att_list = [att_4]
      elif self.node_order == 'k_core':
        adj_list = [adj_5]
        att_list = [att_5]
      elif self.node_order == 'DFS+BFS':
        adj_list = [adj_4, adj_3]
        att_list = [att_4, att_3]
      elif self.node_order == 'DFS+BFS+k_core':
        adj_list = [adj_4, adj_3, adj_5]
        att_list = [att_4, att_3, att_5]
      elif self.node_order == 'DFS+BFS+k_core+degree_decent':
        adj_list = [adj_4, adj_3, adj_5, adj_1]
        att_list = [att_4, att_3, att_5, att_1]
      elif self.node_order == 'all':
        adj_list = [adj_4, adj_3, adj_5, adj_1, adj_0]
        att_list = [att_4, att_3, att_5, att_1, att_0]
      else:
        adj_list = [adj_0]
        att_list = [att_0]

    # print('number of nodes = {}'.format(adj_0.shape[0]))
    return adj_list, att_list

  def __getitem__(self, index):

    print("GET DATA ITEM")
    K = self.block_size
    N = self.max_num_nodes
    S = self.stride

    # load graph
    
    adj_list = pickle.load(open(self.file_names_adj[index], 'rb'))
    print("loaded adj_list:")
    print(adj_list)
    graph = pickle.load(open(self.file_names_adj[index], 'rb'))

    num_nodes = adj_list[0].shape[0]
    num_subgraphs = int(np.floor((num_nodes - K) / S) + 1)

    if self.is_sample_subgraph:
      if self.num_subgraph_batch < num_subgraphs:
        num_subgraphs_pass = int(
            np.floor(self.num_subgraph_batch / self.num_fwd_pass))
      else:
        num_subgraphs_pass = int(np.floor(num_subgraphs / self.num_fwd_pass))

      end_idx = min(num_subgraphs, self.num_subgraph_batch)
    else:
      num_subgraphs_pass = int(np.floor(num_subgraphs / self.num_fwd_pass))
      end_idx = num_subgraphs

    ### random permute subgraph
    rand_perm_idx = self.npr.permutation(num_subgraphs).tolist()

    start_time = time.time()
    data_batch = []
    for ff in range(self.num_fwd_pass):
      ff_idx_start = num_subgraphs_pass * ff
      if ff == self.num_fwd_pass - 1:
        ff_idx_end = end_idx
      else:
        ff_idx_end = (ff + 1) * num_subgraphs_pass

      rand_idx = rand_perm_idx[ff_idx_start:ff_idx_end]

      edges = []
      node_idx_gnn = []
      node_idx_feat = []
      label = []
      subgraph_size = []
      subgraph_idx = []
      att_idx = []
      subgraph_count = 0

      for ii in range(len(adj_list)):
        # loop over different orderings
        adj_full = adj_list[ii]
        # adj_tril = np.tril(adj_full, k=-1)

        idx = -1
        for jj in range(0, num_nodes, S):
          # loop over different subgraphs
          idx += 1

          ### for each size-(jj+K) subgraph, we generate edges for the new block of K nodes
          if jj + K > num_nodes:
            break

          if idx not in rand_idx:
            continue

          ### get graph for GNN propagation
          adj_block = np.pad(
              adj_full[:jj, :jj], ((0, K), (0, K)),
              'constant',
              constant_values=1.0)  # assuming fully connected for the new block
          adj_block = np.tril(adj_block, k=-1)
          adj_block = adj_block + adj_block.transpose()
          adj_block = torch.from_numpy(adj_block).to_sparse()
          print("subgraph adajency", adj_block)
          dense_adj_block = torch.tensor(adj_block).to_dense()
          print("subgraph dense adajency", dense_adj_block)
          print("--------------")
          edges += [adj_block.coalesce().indices().long()]
          print("subgraph edges", edges)
          print("--------------")
          ### get attention index
          # exist node: 0
          # newly added node: 1, ..., K
          if jj == 0:
            att_idx += [np.arange(1, K + 1).astype(np.uint8)]
          else:
            att_idx += [
                np.concatenate([
                    np.zeros(jj).astype(np.uint8),
                    np.arange(1, K + 1).astype(np.uint8)
                ])
            ]
          print("subgraph attion index: ", att_idx)
          ### get node feature index for GNN input
          # use inf to indicate the newly added nodes where input feature is zero
          if jj == 0:
            node_idx_feat += [np.ones(K) * np.inf]
          else:
            node_idx_feat += [
                np.concatenate([np.arange(jj) + ii * N,
                                np.ones(K) * np.inf])
            ]
          print("subgraph node_idx_feat", node_idx_feat)
          print("--------------")
          ### get attention index
          ### get node index for GNN output
          idx_row_gnn, idx_col_gnn = np.meshgrid(
              np.arange(jj, jj + K), np.arange(jj + K))
          idx_row_gnn = idx_row_gnn.reshape(-1, 1)
          idx_col_gnn = idx_col_gnn.reshape(-1, 1)
          node_idx_gnn += [
              np.concatenate([idx_row_gnn, idx_col_gnn],
                             axis=1).astype(np.int64)
          ]
          print("subgraph node_idx_gnn", node_idx_gnn)
          print("--------------")
          ### get predict label
          label += [
              adj_full[idx_row_gnn, idx_col_gnn].flatten().astype(np.uint8)
          ]

          subgraph_size += [jj + K]
          subgraph_idx += [
              np.ones_like(label[-1]).astype(np.int64) * subgraph_count
          ]
          subgraph_count += 1
          print("subgraph label", label)
          print("--------------")
          
      ### adjust index basis for the selected subgraphs
      cum_size = np.cumsum([0] + subgraph_size).astype(np.int64)
      for ii in range(len(edges)):
        edges[ii] = edges[ii] + cum_size[ii]
        node_idx_gnn[ii] = node_idx_gnn[ii] + cum_size[ii]

      graph_attributes = pickle.load(open(self.file_names_att[index], 'rb'))
      
      ### pack tensors
      print("packing tensors:")
      data = {}
      data['attributes'] = np.concatenate(graph_attributes)
      data['adj'] = np.tril(np.stack(adj_list, axis=0), k=-1)
      print("data['adj']: ", data['adj'])
     
      data['edges'] = torch.cat(edges, dim=1).t().long()
      print("data['edges']: ", data['edges'])

      data['node_idx_gnn'] = np.concatenate(node_idx_gnn)
      print("data['node_idx_gnn']: ", data['node_idx_gnn'])

      data['node_idx_feat'] = np.concatenate(node_idx_feat)
      print("data['node_idx_feat']: ", data['node_idx_feat'])

      data['label'] = np.concatenate(label)
      data['att_idx'] = np.concatenate(att_idx)
      data['subgraph_idx'] = np.concatenate(subgraph_idx)
      data['subgraph_count'] = subgraph_count
      data['num_nodes'] = num_nodes
      print("data['num_nodes']: ",  data['num_nodes'])

      data['subgraph_size'] = subgraph_size
      data['num_count'] = sum(subgraph_size)
      data_batch += [data]

    end_time = time.time()

    return data_batch

  def __len__(self):
    return self.num_graphs

  def get_attributes(self, G, order):

    node_attribute_dict = nx.get_node_attributes(G,'label')
    node_attribute_list = [(k,v) for k,v in node_attribute_dict.items()]
    if order == 0:
      attribute_sequence_0 = node_attribute_list
    else:
      print(type(order))
      print(order)
      attribute_sequence_0 = sorted(
          node_attribute_list, key=lambda tt: order.index(tt[0]))
    # Calculate the number of rows (m) and columns (n) for the matrix
    num_rows = len(attribute_sequence_0)
    num_cols = len(attribute_sequence_0[0][1])
    attribute_matrix = [[0] * num_cols for _ in range(num_rows)]
    for i, (_, values) in enumerate(attribute_sequence_0):
      for j, value in enumerate(values):
        attribute_matrix[i][j] = value
    return attribute_matrix

  def collate_fn(self, batch):
    assert isinstance(batch, list)
    start_time = time.time()
    batch_size = len(batch)
    N = self.max_num_nodes
    C = self.num_canonical_order
    batch_data = []

    for ff in range(self.num_fwd_pass):
      data = {}
      batch_pass = []
      for bb in batch:
        batch_pass += [bb[ff]]

      pad_size = [self.max_num_nodes - bb['num_nodes'] for bb in batch_pass]
      subgraph_idx_base = np.array([0] +
                                   [bb['subgraph_count'] for bb in batch_pass])
      subgraph_idx_base = np.cumsum(subgraph_idx_base)

      data['subgraph_idx_base'] = torch.from_numpy(
        subgraph_idx_base)

      data['num_nodes_gt'] = torch.from_numpy(
          np.array([bb['num_nodes'] for bb in batch_pass])).long().view(-1)

      data['adj'] = torch.from_numpy(
          np.stack(
              [
                  np.pad(
                      bb['adj'], ((0, 0), (0, pad_size[ii]), (0, pad_size[ii])),
                      'constant',
                      constant_values=0.0) for ii, bb in enumerate(batch_pass)
              ],
              axis=0)).float()  # B X C X N X N

      idx_base = np.array([0] + [bb['num_count'] for bb in batch_pass])
      idx_base = np.cumsum(idx_base)

      data['edges'] = torch.cat(
          [bb['edges'] + idx_base[ii] for ii, bb in enumerate(batch_pass)],
          dim=0).long()

      data['node_idx_gnn'] = torch.from_numpy(
          np.concatenate(
              [
                  bb['node_idx_gnn'] + idx_base[ii]
                  for ii, bb in enumerate(batch_pass)
              ],
              axis=0)).long()

      data['att_idx'] = torch.from_numpy(
          np.concatenate([bb['att_idx'] for bb in batch_pass], axis=0)).long()

      # shift one position for padding 0-th row feature in the model
      node_idx_feat = np.concatenate(
          [
              bb['node_idx_feat'] + ii * C * N
              for ii, bb in enumerate(batch_pass)
          ],
          axis=0) + 1
      node_idx_feat[np.isinf(node_idx_feat)] = 0
      node_idx_feat = node_idx_feat.astype(np.int64)
      data['node_idx_feat'] = torch.from_numpy(node_idx_feat).long()

      data['label'] = torch.from_numpy(
          np.concatenate([bb['label'] for bb in batch_pass])).float()

      data['subgraph_idx'] = torch.from_numpy(
          np.concatenate([
              bb['subgraph_idx'] + subgraph_idx_base[ii]
              for ii, bb in enumerate(batch_pass)
          ])).long()

      attributes_list = [bb['attributes'] for bb in batch_pass]
    #data['attributes'] =  torch.from_numpy(np.concatenate(attributes_list))
    #Pad each attribute array to have 'max_rows' rows with zeros
      padded_attributes_list = [
          np.pad(arr, ((0, self.max_num_nodes - arr.shape[0]), (0, 0)), 'constant', constant_values=0.0)
          for arr in attributes_list
      ]
      data['attributes'] = torch.from_numpy(np.stack(padded_attributes_list)).float()


      batch_data += [data]

    end_time = time.time()
    #print('collate time = {}'.format(end_time - start_time))

    return batch_data
