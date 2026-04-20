import pickle
import torch
import networkx as nx
from sentence_transformers import SentenceTransformer

class GraphLoader:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)

    def load_and_preprocess(self, path):
        with open(path, 'rb') as f:
            G = pickle.load(f)

        node_names = []
        for n in G.nodes():
            name = G.nodes[n].get('name')
            node_names.append(str(name) if name is not None else "unknown_node")

        x = torch.tensor(self.encoder.encode(node_names), dtype=torch.float)

        node_list = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        edge_list = []
        for u, v in G.edges():
            edge_list.append([node_to_idx[u], node_to_idx[v]])
            edge_list.append([node_to_idx[v], node_to_idx[u]]) 
            
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        return G, x, edge_index, node_list