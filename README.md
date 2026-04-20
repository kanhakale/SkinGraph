# 🧬 Skincare-GNN: Graph Intelligence for Ingredient Safety

An end-to-end **Graph Intelligence** pipeline designed to analyze over 22,000 medical relationships between chemical substances and dermatological conditions. This system utilizes **Sentence Transformers** for semantic encoding and **Graph Convolutional Networks (GCNs)** to identify structural anomalies, rank treatment "Heroes," and flag industrial toxins.

---

## 🚀 Key Features

* **Semantic Graph Construction**: Transforms raw clinical data into a bipartite graph using **NLP embeddings** (384-dimensional vectors).
* **Causal Clustering**: Implements the **Louvain Algorithm** to identify 19 distinct disease-treatment families based on topology.
* **Dual-Layer GNN Audit**:
    * **Node Classification**: Predicts systemic vulnerability of ingredients across the network.
    * **Link Prediction**: Audits the biological plausibility of substance-condition associations with high confidence.
* **Safety Valve Filtering**: Heuristic keyword filtering to prune industrial pollutants (e.g., BPA, Pesticides) from medical recommendations.

---

## 📁 Repository Structure

```text
Skincare-GNN/
├── data/               # Persistent graph storage (.pkl)
├── docs/               # Auto-generated pipeline visualizations (.png)
├── src/                # Core logic
│   ├── models/         # GNN Architectures (Vulnerability & Link Prediction)
│   ├── data_loader.py  # Graph construction & semantic encoding
│   ├── filtering.py    # Heuristic safety logic
│   └── utils.py        # Visualization & metric helpers
├── weights/            # Pre-trained .pth model checkpoints
├── main.py             # Master orchestrator script
└── requirements.txt    # Project dependencies
```

---

## 📊 Pipeline Deliverables

### **D1 & D2: Topology & Clustering**
The system maps the bipartite relationship between **10,352 Substances** and **123 Conditions**. Using Louvain community detection, it uncovers 19 clusters representing specific biological pathologies.
* **Artifacts**: `docs/d1_composition.png`, `docs/d2_clusters.png`

### **D3 & D4: Hero & Redundancy Audit**
Identifies "Hero" treatments based on degree centrality. Notably, the system identifies **1,2-Dimethylhydrazine** (a known carcinogen) as a high-degree "Hero" in raw data—highlighting the necessity for the **GNN Risk Audit** in Deliverable 5.
* **Artifacts**: `docs/hero_results.png`, `docs/d4_redundancy.png`

### **D5: GNN Risk Assessment**
Utilizes a **Graph Convolutional Network (GCN)** architecture to evaluate the network. The link prediction model evaluates associations (e.g., auditing the BPA-Acne connection) by projecting nodes into a latent space and calculating dot-product similarity.
* **Artifact**: `docs/d5_risk_assessment.png`

---

## 🛠️ Installation & Usage

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/Skincare-GNN.git
cd Skincare-GNN

# Install dependencies
pip install -r requirements.txt
```

### 2. Execution
Run the master pipeline to regenerate all analysis and plots:
```bash
python main.py
```

---

## 🧠 Technical Stack

* **Deep Learning**: PyTorch, PyTorch Geometric (**GCNConv**)
* **NLP**: Sentence-Transformers (`all-MiniLM-L6-v2`)
* **Graph Theory**: NetworkX (Louvain, Degree Centrality)
* **Visualization**: Matplotlib, NumPy


---

## 📩 Contact & Contributions

**Authors:** 
Divyank | Priyanshu Kumari | Kanha Shrikant Kale  

*This project was developed as part of an Engineering Research initiative. Contributions, issues, and feature requests are welcome!*

---

## 📝 Disclaimer
This tool is for **research and educational purposes only**. The GNN Risk Assessment is a proof-of-concept model and should not be used as a substitute for professional medical advice or toxicological testing.
---

## ⚖️ License
Distributed under the MIT License.

---
