import os
import sys
import torch
import torch.optim as optim
import numpy as np
from pprint import pprint
from runner import *
from utils.logger import setup_logging
from utils.arg_helper import parse_arguments, get_config
torch.set_printoptions(profile='full')
from model import *
from dataset import *
import pickle
from utils.data_parallel import DataParallel
#import pytest
#import poetry

def load_data(type, num_train, num_dev):

    if type == 'grid_DD':
        graphs = create_graphs(config.dataset.name, data_dir=config.dataset.data_path)
    elif type == 'mesh':
        num_graphs =  100
        graphs = []
        for i in range(0, num_graphs):

            title = "../../artificial_data_gen/data/graph_0.pickle".format(i)
            graphs.append(pickle.load(open(title, 'rb')))

    graphs_train = graphs[:num_train]
    graphs_dev = graphs[:num_dev]
    graphs_test = graphs[num_train:]

    return graphs_train, graphs_dev, graphs_test

def simple_pickle_read(file_name):
  file_path = file_name

  try:
      # Open the file in binary read mode
      with open(file_path, "rb") as file:
          # Load the content of the file using pickle
          data = pickle.load(file)

      return data
  except FileNotFoundError:
      print("File not found.")
      print(file_name)

  except pickle.UnpicklingError as e:
      print("Error while unpickling the file:", e)


if __name__ == "__main__":
    config = get_config('config/gran_DD.yaml', is_test=False)
    print("Hello World!")
  
    graphs_train, graphs_dev, graphs_test = load_data('mesh',5,5)
    train_dataset = eval('GRANData_node_attributes')(config, graphs_train, tag='train')
    
    print("train_dataset: ")
    print(train_dataset.graphs[0].nodes.data())
    print(train_dataset.graphs[0].edges.data())
    print(type(train_dataset))

    print("train_dataset: ")
    print(train_dataset.graphs[3].nodes.data())
    print(train_dataset.graphs[0].edges.data())
    print(type(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= 1, #train_conf.batch_size,
        shuffle=True, #train_conf.shuffle,
        num_workers=2, #train_conf.num_workers,
        collate_fn = train_dataset.collate_fn,
        drop_last=False)
   
    train_iterator = train_loader.__iter__()
    data = next(train_iterator)
    print("after next")
    print("data[0]['node_idx_feat']:")
    print(data[0]['node_idx_feat'])
    print("data[0]['label']:")
    print(data[0]['label'])
    print("data[0]['edges']:")
    print(data[0]['edges'])


    print("eval model: ")
    model = eval('GRANMixtureBernoulli_nodes_and_edges')(config)

   # use_gpu = config.use_gpu
 
    model = DataParallel(model, device_ids=config.gpus).to(config.device)
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=config.train.lr, weight_decay=config.train.wd)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones= config.train.lr_decay_epoch,
        gamma= config.train.lr_decay)
    
    optimizer.zero_grad()

    iter_count = 0    
    results = defaultdict(list)
    max_epoch = 1
    
    for epoch in range(0, max_epoch):
       print("epoch: ", epoch )
       avg_train_loss = .0 
       model.train()
       in_iterator = train_loader.__iter__()
       optimizer.zero_grad()
       num_fwd_pass = 1
       batch_data = []

       data = next(train_iterator)
       batch_data.append(data)
       iter_count += 1
       

       for ff in range(num_fwd_pass):
          batch_fwd = []

          for dd, gpu_id in enumerate(config.gpus):
              data = {}
              data['adj'] = batch_data[dd][ff]['adj'].pin_memory().to(gpu_id, non_blocking=True)          
              data['edges'] = batch_data[dd][ff]['edges'].pin_memory().to(gpu_id, non_blocking=True)
              data['node_idx_gnn'] = batch_data[dd][ff]['node_idx_gnn'].pin_memory().to(gpu_id, non_blocking=True)
              data['node_idx_feat'] = batch_data[dd][ff]['node_idx_feat'].pin_memory().to(gpu_id, non_blocking=True)
              data['label'] = batch_data[dd][ff]['label'].pin_memory().to(gpu_id, non_blocking=True)
              data['att_idx'] = batch_data[dd][ff]['att_idx'].pin_memory().to(gpu_id, non_blocking=True)
              data['subgraph_idx'] = batch_data[dd][ff]['subgraph_idx'].pin_memory().to(gpu_id, non_blocking=True)
              data['subgraph_idx_base'] = batch_data[dd][ff]['subgraph_idx_base'].pin_memory().to(gpu_id, non_blocking=True)
              data['attributes'] = batch_data[dd][ff]['attributes'].pin_memory().to(gpu_id, non_blocking=True)
              if dd == 0:
                print("training_data: " )
                print("'adj']: ", data['adj'].shape)
                print("['edges]: ", data['edges'])
                print("'att_idx:",  data['att_idx'] )
                print("node_idx_feat: ",  data['node_idx_feat'])
                print("node_idx_gnn: ",  data['node_idx_gnn'])
                print("label: ", data['label'])
                print('attributes: ', data['attributes'].shape)
                print('subgraph_idx: ', data['subgraph_idx'])
                print('subgraph_idx_base: ', data['subgraph_idx_base'])

              batch_fwd.append((data,))
          
          
          
          if batch_fwd:
              train_loss = model(*batch_fwd).mean()   
              print("train_loss: ")
              print(train_loss)           
              avg_train_loss += train_loss              
              # assign gradient
              train_loss.backward()

       optimizer.step()
       lr_scheduler.step()
       avg_train_loss /= float(num_fwd_pass)
       print("avg_train_loss: ")
       print(avg_train_loss)
       # reduce
       avg_train_loss = avg_train_loss.clone().detach()
       train_loss = float(avg_train_loss.data.cpu().numpy())
       print("train_loss: ")
       print(train_loss)

    # clip_grad_norm_(model.parameters(), 5.0e-0)

    optimizer.step()
    avg_train_loss /= float(num_fwd_pass)   