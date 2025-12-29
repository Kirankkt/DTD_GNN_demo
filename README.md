---
title: DTD-GNN Drug Repurposing
emoji: 💊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# DTD-GNN Drug Repurposing Platform

Discover potential new uses for existing drugs using Graph Neural Networks.

## Model Performance
- **AUC: 0.9987** (State-of-the-art)
- **AUPR: 0.9986**
- Trained on 300,000 drug-target-disease events

## Features
- Find drug candidates for any disease
- Find potential new uses for any drug
- Validate specific Drug-Target-Disease predictions
- Download results as CSV

## Architecture
- **HeteroTransformerGNN** with 8-head attention
- Skip connections for improved gradient flow
- Bidirectional message passing on heterogeneous graph
- MLP link predictor

## Data
- 5,322 drugs (DrugBank)
- 2,844 protein targets (UniProt)
- 6,055 diseases (MeSH)
- Based on BioSNAP dataset
