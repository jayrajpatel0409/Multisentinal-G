# MultiSentinel-G: Opinion-Spam Coordination Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Research Paper Implementation & Interactive Evaluation Framework**  
> *"How Much Do Coordination Graphs Add to Opinion-Spam Detection Beyond Reviewer Context?"*

---

## 📌 Executive Summary

Modern opinion-spam operations no longer rely on single accounts copying and pasting text. Instead, coordinated botnets spread reviews across multiple accounts to evade platform filters. While existing literature claims that Graph Neural Networks (GNNs) achieve massive improvements over text classifiers, our research investigates a fundamental scientific control:

> **Does the network graph actually catch coordinated collusion, or is it mostly just a roundabout proxy for the reviewer's personal account history?**

### Key Findings on 960,000+ Real Yelp Reviews (YelpZip & YelpNYC):
1. **Text Classifiers Alone Fail (~0.29 AP)**: Attackers easily bypass NLP models by paraphrasing or generating realistic text with AI.
2. **Account History Does the Heavy Lifting (~0.48 AP)**: Reviewer-level context (posting velocity, account age, rating polarity) accounts for **92% to 97%** of the performance boost previously attributed to complex graph models.
3. **MultiSentinel-G Catches Coordinated Collusion**: Achieves competitive detection accuracy without requiring heavy GPU deep learning, while protecting real-world social groups (e.g. friends dining together) from false-positive bans.

---

## 🚀 Quick Start: Interactive Research Dashboard

The repository includes a full-featured **Streamlit research dashboard** with interactive network graphs, real-world case studies, and live inference testing.

### 1. Launch with One Click (Windows)
Double-click:
```bash
run_dashboard.bat
```

### 2. Manual Setup & Launch
```bash
# Clone the repository
git clone https://github.com/jayrajpatel0409/Multisentinal-G.git
cd Multisentinal-G

# Install dependencies
pip install -r code_archive/requirements.txt
pip install streamlit plotly networkx

# Run the research dashboard
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🗂️ Repository Structure

```text
├── app.py                      # Main Streamlit research & evaluation dashboard
├── app_data.py                 # Case study provider and live inference engine
├── demo_live.py                # Command-line spot-check script (~7.8s runtime)
├── run_dashboard.bat           # 1-click Windows dashboard launcher
├── MultiSentinel_G_manuscript.pdf # Full written research manuscript
│
├── result_data/                # Derived benchmark datasets (10 CSV files)
│   ├── d01_yelpzip_main_results_per_seed.csv
│   ├── d02_yelpnyc_main_results_per_seed.csv
│   ├── d03_account_context_control_yelpzip.csv
│   ├── d08_friend_group_test_yelp_open.csv
│   └── d10_campaignsim_simulation_results.csv
│
└── code_archive/               # Core algorithm implementations
    ├── code/
    │   ├── coordgraph.py       # Multiplex coordination graph builder
    │   ├── campaignsim.py      # Adversarial campaign simulation engine
    │   ├── pipeline.py         # 5-fold cross-validation pipeline
    │   └── yelp_nested.py      # Real-data evaluation runner
    └── requirements.txt        # Core package requirements
```

---

## 📊 Core Benchmark Comparison

Evaluated under strict **Reviewer-Grouped 5-Fold Cross-Validation** (zero test-set leakage):

| Detection Methodology | YelpZip (AP) | YelpNYC (AP) | Key Finding |
|---|---|---|---|
| **1. Text-Only Classifier** | 0.293 | 0.226 | Misses bots that write natural phrasing |
| **2. Text + Basic Metadata** | 0.465 | 0.365 | Catches obvious rapid single posters |
| **3. Reviewer Account History** | **0.478** | **0.384** | **Massive jump**: captures chronic spammers |
| **4. MultiSentinel-G (Proposed)** | **0.479** | **0.386** | **Catches coordinated bot rings without GPU training** |
| **5. GraphSAGE GNN (Deep Learning)** | 0.491 | 0.400 | Heavy GPU compute cost for marginal +0.01 gain |

---

## 👥 Confounder Analysis: Friends vs. Fraud

On the **Yelp Open Dataset** (which includes real social friendship graphs), we discovered that **25.5% of raw co-review timing links join real-world friends** (93.2× higher than random chance). MultiSentinel-G integrates degree-corrected null filtering to ensure legitimate diners dining together are never penalized.

---

## 📄 License & Attribution
The derived benchmark datasets and code are released under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
