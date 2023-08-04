from utils.data_helper import *
import pickle

graphs = create_graphs('FIRSTMM_DB', data_dir='data/')
print("GRAN graph")
print(*graphs[0])
graph = graphs[0]
print(graph.nodes.data())

num_graphs =  100
vs_graphs = []
for i in range(0, num_graphs):#
        title = "../../artificial_data_gen/graph_{}.pickle".format(i)
        vs_graphs.append(pickle.load(open(title, 'rb')))
print("vs synthetic graph")
print(vs_graphs[0].nodes.data())
# Get node positions for the 3D plot
#vs_graph = vs_graphs[0]
#pos = {node: vs_graph.nodes[node]["pos"] for node in vs_graph.nodes}
#print(pos)