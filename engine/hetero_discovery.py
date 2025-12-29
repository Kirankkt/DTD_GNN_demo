"""
Heterogeneous Discovery Engine for Drug Repurposing

This module enables actual drug repurposing predictions using the trained
HeteroTransformerGNN model. It can:
1. Find drugs for a given disease
2. Find diseases a drug might treat
3. Predict novel Drug-Target-Disease triplets
4. Validate predictions against known repurposing cases
"""
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from engine.hetero_models import HeteroTransformerGNN, HeteroLinkPredictor


class HeteroDiscoveryEngine:
    """
    Drug repurposing discovery engine using the trained HeteroTransformerGNN.

    Usage:
        engine = HeteroDiscoveryEngine("data/dtd_full_checkpoint.pt", "data/dtd_full_graph.pt")
        results = engine.find_drugs_for_disease("MESH:D000544")  # Alzheimer's
    """

    def __init__(self, model_path: str, data_path: str, device: str = None):
        """
        Initialize the discovery engine.

        Args:
            model_path: Path to trained model checkpoint
            data_path: Path to saved graph data
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        print(f"Discovery Engine using device: {self.device}")

        # Load graph data
        print("Loading graph data...")
        self.data = torch.load(data_path, map_location=self.device, weights_only=False)
        self.data = self.data.to(self.device)

        # Extract mappings
        self.drug_names = getattr(self.data, 'drug_names', {})
        self.target_names = getattr(self.data, 'target_names', {})
        self.disease_names = getattr(self.data, 'disease_names', {})
        self.events = getattr(self.data, 'events', [])

        # Build reverse mappings
        self.drug_to_id = {v: k for k, v in self.drug_names.items()}
        self.target_to_id = {v: k for k, v in self.target_names.items()}
        self.disease_to_id = {v: k for k, v in self.disease_names.items()}

        # Build event lookup
        self.event_by_id = {e['id']: e for e in self.events}

        # Load model
        print("Loading model...")
        self.gnn = HeteroTransformerGNN(
            in_channels=1024,
            hidden_channels=256,
            out_channels=128
        ).to(self.device)

        self.predictor = HeteroLinkPredictor(in_channels=128).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.gnn.load_state_dict(checkpoint['gnn_state_dict'])
        self.predictor.load_state_dict(checkpoint['predictor_state_dict'])

        self.gnn.eval()
        self.predictor.eval()

        # Pre-compute embeddings
        print("Computing embeddings...")
        self._compute_embeddings()

        print(f"Discovery Engine ready!")
        print(f"  - {len(self.drug_names)} drugs")
        print(f"  - {len(self.target_names)} targets")
        print(f"  - {len(self.disease_names)} diseases")
        print(f"  - {len(self.events)} events")

    def _compute_embeddings(self):
        """Pre-compute all node embeddings for fast inference."""
        with torch.no_grad():
            x_dict = {
                'drug': self.data['drug'].x,
                'target': self.data['target'].x,
                'disease': self.data['disease'].x,
                'event': self.data['event'].x,
            }
            self.z_dict = self.gnn(x_dict, self.data.edge_index_dict)

    def find_drugs_for_disease(self, disease_id: str, top_k: int = 20) -> List[Dict]:
        """
        Find potential drugs for treating a disease.

        This works by finding events (Drug-Target combinations) that have
        high predicted probability of treating the disease.

        Args:
            disease_id: MESH ID of the disease (e.g., "MESH:D000544")
            top_k: Number of top candidates to return

        Returns:
            List of dictionaries with drug, target, and score information
        """
        if disease_id not in self.disease_to_id:
            raise ValueError(f"Disease '{disease_id}' not found. Try searching with search_diseases()")

        disease_idx = self.disease_to_id[disease_id]

        with torch.no_grad():
            z_disease = self.z_dict['disease'][disease_idx].unsqueeze(0)
            z_disease = z_disease.expand(self.z_dict['event'].size(0), -1)

            # Score all events against this disease
            scores = self.predictor(self.z_dict['event'], z_disease)

            # Get top-k events
            top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))

        results = []
        for score, event_idx in zip(top_scores.cpu().numpy(), top_indices.cpu().numpy()):
            event = self.event_by_id.get(int(event_idx), {})
            results.append({
                'event_id': int(event_idx),
                'drug': event.get('drug', 'Unknown'),
                'target': event.get('target', 'Unknown'),
                'disease': disease_id,
                'score': float(score),
                'confidence': 'High' if score > 0.9 else 'Medium' if score > 0.7 else 'Low'
            })

        return results

    def find_diseases_for_drug(self, drug_id: str, top_k: int = 20) -> List[Dict]:
        """
        Find potential diseases that a drug might treat.

        Args:
            drug_id: DrugBank ID (e.g., "DB00001")
            top_k: Number of top candidates to return

        Returns:
            List of dictionaries with disease and score information
        """
        if drug_id not in self.drug_to_id:
            raise ValueError(f"Drug '{drug_id}' not found. Try search_drugs()")

        # Find all events involving this drug
        drug_events = [e for e in self.events if e['drug'] == drug_id]

        if not drug_events:
            return []

        # Score these events against all diseases
        results = []

        with torch.no_grad():
            for event in drug_events:
                event_idx = event['id']
                z_event = self.z_dict['event'][event_idx].unsqueeze(0)
                z_event = z_event.expand(self.z_dict['disease'].size(0), -1)

                scores = self.predictor(z_event, self.z_dict['disease'])

                # Get top diseases for this event
                top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))

                for score, disease_idx in zip(top_scores.cpu().numpy(), top_indices.cpu().numpy()):
                    disease_name = self.disease_names.get(int(disease_idx), 'Unknown')
                    results.append({
                        'drug': drug_id,
                        'target': event['target'],
                        'disease': disease_name,
                        'score': float(score),
                        'event_id': event_idx
                    })

        # Sort by score and deduplicate by disease
        results.sort(key=lambda x: x['score'], reverse=True)
        seen_diseases = set()
        unique_results = []
        for r in results:
            if r['disease'] not in seen_diseases:
                seen_diseases.add(r['disease'])
                unique_results.append(r)
                if len(unique_results) >= top_k:
                    break

        return unique_results

    def predict_triplet(self, drug_id: str, target_id: str, disease_id: str) -> Dict:
        """
        Predict the likelihood that a specific Drug-Target-Disease triplet is valid.

        Args:
            drug_id: DrugBank ID
            target_id: UniProt ID
            disease_id: MESH ID

        Returns:
            Dictionary with prediction score and details
        """
        # Find matching event
        matching_events = [
            e for e in self.events
            if e['drug'] == drug_id and e['target'] == target_id
        ]

        if not matching_events:
            return {
                'valid': False,
                'message': f"No event found for drug {drug_id} and target {target_id}",
                'score': None
            }

        if disease_id not in self.disease_to_id:
            return {
                'valid': False,
                'message': f"Disease {disease_id} not found",
                'score': None
            }

        event = matching_events[0]
        disease_idx = self.disease_to_id[disease_id]

        with torch.no_grad():
            z_event = self.z_dict['event'][event['id']]
            z_disease = self.z_dict['disease'][disease_idx]
            score = self.predictor(z_event.unsqueeze(0), z_disease.unsqueeze(0))

        return {
            'valid': True,
            'drug': drug_id,
            'target': target_id,
            'disease': disease_id,
            'score': float(score.item()),
            'prediction': 'Likely' if score > 0.5 else 'Unlikely',
            'confidence': 'High' if score > 0.9 else 'Medium' if score > 0.7 else 'Low'
        }

    def search_drugs(self, query: str) -> List[str]:
        """Search for drugs by partial ID match."""
        query = query.upper()
        return [d for d in self.drug_to_id.keys() if query in d.upper()]

    def search_diseases(self, query: str) -> List[str]:
        """Search for diseases by partial ID match."""
        query = query.upper()
        return [d for d in self.disease_to_id.keys() if query in d.upper()]

    def search_targets(self, query: str) -> List[str]:
        """Search for targets by partial ID match."""
        query = query.upper()
        return [t for t in self.target_to_id.keys() if query in t.upper()]

    def get_statistics(self) -> Dict:
        """Get summary statistics about the model and data."""
        return {
            'num_drugs': len(self.drug_names),
            'num_targets': len(self.target_names),
            'num_diseases': len(self.disease_names),
            'num_events': len(self.events),
            'device': str(self.device),
            'model_params': sum(p.numel() for p in self.gnn.parameters()),
        }

    def validate_known_repurposing(self, known_cases: List[Dict]) -> pd.DataFrame:
        """
        Validate the model against known drug repurposing cases.

        Args:
            known_cases: List of dicts with 'drug', 'target', 'disease' keys

        Returns:
            DataFrame with validation results
        """
        results = []

        for case in known_cases:
            pred = self.predict_triplet(
                case['drug'],
                case['target'],
                case['disease']
            )
            results.append({
                'drug': case['drug'],
                'target': case['target'],
                'disease': case['disease'],
                'expected': case.get('expected', 'positive'),
                'score': pred.get('score'),
                'prediction': pred.get('prediction'),
                'valid': pred.get('valid')
            })

        return pd.DataFrame(results)


# Known drug repurposing cases for validation
KNOWN_REPURPOSING_CASES = [
    # Sildenafil (Viagra) - originally for angina, repurposed for erectile dysfunction
    {'drug': 'DB00203', 'target': 'P00451', 'disease': 'MESH:D007172', 'expected': 'positive'},

    # Thalidomide - originally sedative, repurposed for multiple myeloma
    {'drug': 'DB01041', 'target': 'P35354', 'disease': 'MESH:D009101', 'expected': 'positive'},

    # Minoxidil - originally hypertension, repurposed for hair loss
    {'drug': 'DB00350', 'target': 'P19634', 'disease': 'MESH:D000505', 'expected': 'positive'},

    # Aspirin - pain relief, repurposed for cardiovascular protection
    {'drug': 'DB00945', 'target': 'P23219', 'disease': 'MESH:D002318', 'expected': 'positive'},
]


def demo():
    """Demo the discovery engine."""
    print("="*70)
    print("DTD-GNN Drug Repurposing Discovery Engine - Demo")
    print("="*70)

    try:
        engine = HeteroDiscoveryEngine(
            model_path="data/dtd_full_checkpoint.pt",
            data_path="data/dtd_full_graph.pt"
        )
    except FileNotFoundError:
        print("\nModel not found! Please train first with:")
        print("  python main_dtd.py --epochs 100")
        return

    print("\n" + "="*70)
    print("Statistics")
    print("="*70)
    stats = engine.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Find a disease to query
    print("\n" + "="*70)
    print("Finding drugs for a disease...")
    print("="*70)

    # Pick first disease
    if engine.disease_to_id:
        sample_disease = list(engine.disease_to_id.keys())[0]
        print(f"\nTop drugs for {sample_disease}:")

        results = engine.find_drugs_for_disease(sample_disease, top_k=10)
        for i, r in enumerate(results[:10], 1):
            print(f"  {i}. Drug: {r['drug']}, Target: {r['target']}, "
                  f"Score: {r['score']:.4f} ({r['confidence']})")

    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)


if __name__ == "__main__":
    demo()
