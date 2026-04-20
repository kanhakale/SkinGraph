import torch
import networkx as nx
from networkx.algorithms import community
from src.data_loader import GraphLoader
from src.filter import SemanticFilter
from src.models import VulnerabilityModel, Side_EffectModel
from src.utils import plot_bipartite_balance,plot_gnn_results,plot_cluster_sizes,plot_redundancy_distribution, plot_hero_potency, format_confidence

def run_full_pipeline(graph_path, model_1_path, model_2_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Full Skincare Intelligence Pipeline on {device}")
    print('\n')

    print("\n [D1] Initializing Network Construction...")
    print('\n')
    loader = GraphLoader()
    G, x, edge_index, node_list = loader.load_and_preprocess(graph_path)
    substances = [n for n in G.nodes() if G.nodes[n].get('label') == 'Substance']
    conditions = [n for n in G.nodes() if G.nodes[n].get('label') == 'Condition']

    if len(substances) == 0:
        print("ERROR: List is empty. Check the labels once maybe")
        return
    print(f"Bipartite Graph Created: {len(substances)} Substances | {len(conditions)} Conditions")
    plot_bipartite_balance(len(substances), len(conditions), save_path='docs/d1_composition.png')
    print("D1 Plot saved to docs/d1_composition.png")
    print('\n')

    print("\n[D2] Performing Causal Clustering (Louvain)...")
    communities = community.louvain_communities(G, seed=42)
    print(f"Identified {len(communities)} distinct Causal Clusters in the network.")
    plot_cluster_sizes(communities, save_path='docs/d2_clusters.png')
    print("D2 Plot saved to docs/d2_clusters.png")
    print('\n')

    print("\n[D3] Identifying Hero Treatments...")
    raw_rankings = [(G.nodes[n].get('name'), G.degree(n)) for n in substances]
    raw_rankings.sort(key=lambda x: x[1], reverse=True)
    sf = SemanticFilter()
    safe_heroes = sf.filter_hero_list(raw_rankings)
    print(f"🏆 Top Hero Found: {safe_heroes[0][0]} treating {safe_heroes[0][1]} conditions.")
    plot_hero_potency(safe_heroes, top_n=10, save_path='docs/hero_results.png')
    print('\n')

    print("\n[D4] Analyzing Treatment Redundancy...")
    redundancy = {G.nodes[n].get('name'): G.degree(n) for n in conditions}
    brittle_nodes = [name for name, deg in redundancy.items() if deg == 1]
    print(f"Redundancy Audit: Found {len(brittle_nodes)} 'Brittle' conditions (Redundancy = 1).")
    print("\n[D4] Analyzing Treatment Redundancy...")
    redundancy_counts = [G.degree(n) for n in conditions]
    plot_redundancy_distribution(redundancy_counts, save_path='docs/d4_redundancy.png')
    
    brittle_nodes = [n for n in conditions if G.degree(n) == 1]
    print(f"D4 Plot saved to docs/d4_redundancy.png")
    print('\n')
    print("\n[D5] Running GNN Risk Assessment...")
    
    print("====> Loading Model 1: Vulnerability Classifier...")
    model_1 = VulnerabilityModel().to(device)

    checkpoint1 = torch.load(model_1_path, map_location=device)

    if isinstance(checkpoint1, dict) and 'model_state_dict' in checkpoint1:
        state_dict1 = checkpoint1['model_state_dict']
    else:
        state_dict1 = checkpoint1

    current_model_dict = model_1.state_dict()
    new_state_dict = {}

    for k, v in state_dict1.items():
        if k.endswith('.weight') and k.replace('.weight', '.lin.weight') in current_model_dict:
            new_state_dict[k.replace('.weight', '.lin.weight')] = v
        elif k.endswith('.bias') and k.replace('.bias', '.lin.bias') in current_model_dict:
            new_state_dict[k.replace('.bias', '.lin.bias')] = v
        else:
            new_state_dict[k] = v

    model_1.load_state_dict(new_state_dict, strict=False)
    model_1.eval()
    print("Model 1 loaded and remapped")
    print('\n')
    
    with torch.no_grad():
        out = model_1(x.to(device), edge_index.to(device))
        preds = out.argmax(dim=1)
        vulnerable_count = (preds == 1).sum().item()
        print(f"Model 1 Result: AI flagged {vulnerable_count} nodes as systemically vulnerable.")

    vulnerable_count = (preds == 1).sum().item()
    safe_count = (preds == 0).sum().item()
    
    plot_gnn_results(vulnerable_count, safe_count, save_path='docs/d5_risk_assessment.png')
    print(f"D5 Plot saved to docs/d5_risk_assessment.png")
    print('\n')

    print("   =======> Loading Model 2: Link Prediction Auditor...")
    model_2 = Side_EffectModel().to(device)
    checkpoint2 = torch.load(model_2_path, map_location=device)
    state_dict2 = checkpoint2['model_state_dict'] if 'model_state_dict' in checkpoint2 else checkpoint2
    model_2.load_state_dict(state_dict2)
    model_2.eval()

    with torch.no_grad():
        z = model_2.encode(x.to(device), edge_index.to(device))
        def get_idx(name):
            return next(i for i, n in enumerate(node_list) if str(G.nodes[n].get('name')).lower() == name.lower())
        try:
            s_idx, c_idx = get_idx('bisphenol A'), get_idx('Acne, Adult')
            prob = torch.sigmoid((z[s_idx] * z[c_idx]).sum(dim=-1)).item()
            print(f"Model 2 Stress Test: BPA -> Adult Acne | Confidence: {format_confidence(prob)}")
        except StopIteration:
            print("Stress test target not found in current graph slice.")
    print('\n')

    print("\n Full Pipeline Execution Complete. Results saved to /docs.")

if __name__ == "__main__":
    GRAPH_PATH = r'data\Derm_Graph.pkl'
    MODEL_1_WTS = 'weights/vulnerability_model.pth'
    MODEL_2_WTS = 'weights/side_effect_model.pth'
    
    run_full_pipeline(GRAPH_PATH, MODEL_1_WTS, MODEL_2_WTS)