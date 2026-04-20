import matplotlib.pyplot as plt
import numpy as np

def plot_bipartite_balance(substance_count, condition_count, save_path=None):
    labels = ['Substances', 'Conditions']
    counts = [substance_count, condition_count]
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts, color=['#1abc9c', '#9b59b6'], alpha=0.8)
    plt.title("D1: Bipartite Network Composition")
    plt.ylabel("Node Count")
    for i, v in enumerate(counts):
        plt.text(i, v + 100, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.close()

def plot_cluster_sizes(communities, save_path=None):
    sizes = sorted([len(c) for c in communities], reverse=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(sizes) + 1), sizes, marker='o', linestyle='-', color='#e67e22')
    plt.fill_between(range(1, len(sizes) + 1), sizes, color='#e67e22', alpha=0.2)
    plt.title("D2: Causal Cluster Size Distribution")
    plt.xlabel("Cluster ID (Ranked)")
    plt.ylabel("Number of Nodes in Cluster")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.close()

def plot_hero_potency(hero_data, top_n=10, save_path=None):
    names, scores = zip(*hero_data[:top_n])
    plt.figure(figsize=(12, 6))
    plt.bar(names, scores, color='#f1c40f')
    plt.title(f"D3: Top {top_n} Treatment Influence (Hero Potency)")
    plt.ylabel("Number of Conditions Treated")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.close()

def plot_redundancy_distribution(redundancy_data, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.hist(redundancy_data, bins=range(1, 20), color='#3498db', edgecolor='black', alpha=0.7)
    plt.title("D4: Treatment Redundancy Distribution")
    plt.xlabel("Number of Available Treatments")
    plt.ylabel("Number of Conditions")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.close()

def plot_gnn_results(vulnerable_count, safe_count, save_path=None):
    labels = ['Safe/Stable', 'Vulnerable']
    sizes = [safe_count, vulnerable_count]
    colors = ['#2ecc71', '#e74c3c']
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, explode=(0, 0.1))
    plt.title("D5: GNN Vulnerability Classification")
    if save_path: plt.savefig(save_path)
    plt.close()

def format_confidence(prob):
    return f"{prob * 100:.2f}% Confidence"