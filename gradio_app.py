"""
DTD-GNN Drug Repurposing Platform - Gradio Version

Run in Colab:
    !pip install gradio torch torch-geometric pandas numpy scikit-learn
    !python gradio_app.py
"""
import gradio as gr
import torch
import pandas as pd
import os

# Disease name mappings
DISEASE_NAMES = {
    "MESH:D000544": "Alzheimer's Disease",
    "MESH:D001943": "Breast Cancer",
    "MESH:D003920": "Diabetes Mellitus",
    "MESH:D006973": "Hypertension",
    "MESH:D003324": "Coronary Artery Disease",
    "MESH:D001249": "Asthma",
    "MESH:D012559": "Schizophrenia",
    "MESH:D003863": "Depression",
    "MESH:D010300": "Parkinson's Disease",
    "MESH:D008175": "Lung Cancer",
    "MESH:D015179": "Colorectal Cancer",
    "MESH:D008545": "Melanoma",
    "MESH:D009369": "Neoplasms (Cancer)",
    "MESH:D001172": "Rheumatoid Arthritis",
    "MESH:D015658": "HIV/AIDS",
    "MESH:D008180": "Lupus (SLE)",
    "MESH:D009103": "Multiple Sclerosis",
    "MESH:D004827": "Epilepsy",
    "MESH:D009765": "Obesity",
    "MESH:D008288": "Malaria",
    "MESH:D014376": "Tuberculosis",
    "MESH:D006333": "Heart Failure",
    "MESH:D020521": "Stroke",
    "MESH:D007676": "Kidney Disease",
    "MESH:D008107": "Liver Disease",
    "MESH:D011471": "Prostate Cancer",
    "MESH:D010051": "Ovarian Cancer",
    "MESH:D007938": "Leukemia",
    "MESH:D008223": "Lymphoma",
    "MESH:D012595": "Scleroderma",
}

DRUG_NAMES = {
    "DB00945": "Aspirin",
    "DB00316": "Acetaminophen",
    "DB00472": "Fluoxetine (Prozac)",
    "DB00215": "Citalopram",
    "DB01050": "Ibuprofen",
    "DB00635": "Prednisone",
    "DB00563": "Methotrexate",
    "DB00619": "Imatinib (Gleevec)",
    "DB00331": "Metformin",
    "DB00203": "Sildenafil (Viagra)",
    "DB00675": "Tamoxifen",
    "DB00515": "Cisplatin",
    "DB00997": "Doxorubicin",
    "DB00441": "Gemcitabine",
    "DB00958": "Carboplatin",
    "DB01229": "Paclitaxel",
    "DB00264": "Metoprolol",
    "DB00571": "Propranolol",
    "DB00722": "Lisinopril",
    "DB00678": "Losartan",
    "DB00177": "Valsartan",
    "DB00721": "Memantine (Alzheimer's)",
    "DB00920": "Ketotifen (Antihistamine)",
    "DB00714": "Apomorphine (Parkinson's)",
    "DB01048": "Abacavir (HIV)",
    "DB00682": "Warfarin (Blood thinner)",
    "DB01185": "Fluoxymesterone",
    "DB03127": "Flavin adenine dinucleotide",
    "DB00238": "Nevirapine (HIV)",
    "DB00388": "Phenylephrine",
    "DB00231": "Temazepam (Sleep)",
    "DB01236": "Sevoflurane (Anesthetic)",
    "DB01174": "Phenobarbital (Seizures)",
    "DB02740": "Aminopterin",
    "DB01234": "Dexamethasone",
    "DB00343": "Diltiazem (Heart)",
    "DB00503": "Ritonavir (HIV)",
    "DB00761": "Potassium chloride",
    "DB00289": "Atomoxetine (ADHD)",
    "DB00477": "Chlorpromazine (Antipsychotic)",
    "DB00679": "Thioridazine",
    "DB00763": "Methimazole (Thyroid)",
    "DB00250": "Dapsone (Leprosy)",
    "DB00067": "Vasopressin",
    "DB00584": "Enalapril (Blood pressure)",
    "DB00711": "Diethylcarbamazine",
    "DB00696": "Ergotamine (Migraine)",
    "DB01169": "Arsenic trioxide (Leukemia)",
    "DB00392": "Profenamine",
    "DB00709": "Lamivudine (HIV)",
    "DB00201": "Caffeine",
    "DB03575": "Oxalic acid",
    "DB01244": "Bepridil",
    "DB00659": "Acamprosate (Alcoholism)",
    "DB00363": "Clozapine (Schizophrenia)",
    "DB00248": "Cabergoline (Parkinson's)",
    "DB00974": "Edrophonium",
    "DB00369": "Cidofovir (Antiviral)",
    "DB00883": "Isosorbide dinitrate (Heart)",
    "DB02772": "4-Aminobutyric acid",
    "DB00242": "Cladribine (Cancer)",
    "DB00458": "Imipramine (Depression)",
    "DB00653": "Magnesium sulfate",
    "DB01259": "Lapatinib (Cancer)",
    "DB00784": "Mefenamic acid (Pain)",
    "DB00929": "Misoprostol (Ulcer)",
}

TARGET_NAMES = {
    "P35354": "COX-2 (Prostaglandin synthase)",
    "P23219": "COX-1",
    "P08684": "CYP3A4 (Drug metabolism)",
    "P10635": "CYP2D6 (Drug metabolism)",
    "P00533": "EGFR (Cancer target)",
    "P04626": "HER2 (Breast cancer)",
    "P00519": "ABL1 (Leukemia target)",
    "P42345": "mTOR (Cancer target)",
    "P04798": "CYP1A1",
    "Q16678": "CYP1B1",
}


def get_name(id_val, name_dict):
    """Get human-readable name or return ID."""
    return name_dict.get(id_val, id_val)


# Global engine variable
engine = None


def load_engine():
    """Load the discovery engine."""
    global engine
    if engine is not None:
        return engine

    from engine.hetero_discovery import HeteroDiscoveryEngine

    model_path = "data/dtd_full_checkpoint.pt"
    data_path = "data/dtd_full_graph.pt"

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        return None

    engine = HeteroDiscoveryEngine(model_path, data_path)
    return engine


def get_disease_choices():
    """Get list of diseases for dropdown."""
    eng = load_engine()
    if eng is None:
        return ["Model not loaded"]

    choices = []
    for mesh_id in eng.disease_to_id.keys():
        name = DISEASE_NAMES.get(mesh_id, mesh_id)
        choices.append(f"{name} ({mesh_id})")

    # Sort with known diseases first
    known = [c for c in choices if not c.startswith("MESH:")]
    unknown = [c for c in choices if c.startswith("MESH:")]
    return sorted(known) + sorted(unknown)


def find_drugs_for_disease(disease_selection, top_k):
    """Find drug candidates for a disease."""
    eng = load_engine()
    if eng is None:
        return "Error: Model not loaded. Please ensure model files are in data/ folder."

    # Parse disease ID from selection
    mesh_id = disease_selection.split("(")[-1].rstrip(")")

    results = eng.find_drugs_for_disease(mesh_id, top_k=int(top_k))

    if not results:
        return "No results found."

    # Format as markdown table
    output = f"## Found {len(results)} drug candidates for {disease_selection}\n\n"
    output += "| Rank | Drug | Drug ID | Target | Target ID | Score | Confidence |\n"
    output += "|------|------|---------|--------|-----------|-------|------------|\n"

    for i, r in enumerate(results, 1):
        drug_name = get_name(r['drug'], DRUG_NAMES)
        target_name = get_name(r['target'], TARGET_NAMES)
        output += f"| {i} | {drug_name} | {r['drug']} | {target_name} | {r['target']} | {r['score']:.4f} | {r['confidence']} |\n"

    output += "\n\n### Look up IDs:\n"
    output += "- **Drug IDs (DB...)**: [DrugBank](https://go.drugbank.com/drugs/) - add ID to URL\n"
    output += "- **Target IDs (P.../Q...)**: [UniProt](https://www.uniprot.org/uniprotkb/) - add ID to URL\n"
    output += "- **Disease IDs (MESH:D...)**: [MeSH](https://meshb.nlm.nih.gov/record/ui?ui=) - add ID after ui=\n"

    return output


def find_diseases_for_drug(drug_id, top_k):
    """Find potential diseases for a drug."""
    eng = load_engine()
    if eng is None:
        return "Error: Model not loaded."

    matches = eng.search_drugs(drug_id.strip())

    if not matches:
        return f"Drug '{drug_id}' not found in database."

    results = eng.find_diseases_for_drug(matches[0], top_k=int(top_k))

    if not results:
        return "No results found."

    drug_name = get_name(matches[0], DRUG_NAMES)

    output = f"## Found {len(results)} potential uses for {drug_name} ({matches[0]})\n\n"
    output += "| Rank | Disease | Disease ID | Target | Score |\n"
    output += "|------|---------|------------|--------|-------|\n"

    for i, r in enumerate(results, 1):
        disease_name = get_name(r['disease'], DISEASE_NAMES)
        target_name = get_name(r['target'], TARGET_NAMES)
        output += f"| {i} | {disease_name} | {r['disease']} | {target_name} | {r['score']:.4f} |\n"

    return output


def validate_prediction(drug_id, target_id, disease_id):
    """Validate a specific Drug-Target-Disease prediction."""
    eng = load_engine()
    if eng is None:
        return "Error: Model not loaded."

    result = eng.predict_triplet(drug_id.strip(), target_id.strip(), disease_id.strip())

    if not result['valid']:
        return f"Could not validate: {result.get('message', 'Unknown error')}"

    score = result['score']
    drug_name = get_name(drug_id.strip(), DRUG_NAMES)
    disease_name = get_name(disease_id.strip(), DISEASE_NAMES)
    target_name = get_name(target_id.strip(), TARGET_NAMES)

    if score > 0.9:
        confidence = "HIGH"
        emoji = "+"
    elif score > 0.7:
        confidence = "MEDIUM"
        emoji = "~"
    else:
        confidence = "LOW"
        emoji = "-"

    output = f"## {confidence} Confidence Prediction {emoji}\n\n"
    output += f"**Score:** {score:.4f}\n\n"
    output += f"**Drug:** {drug_name} ({drug_id})\n\n"
    output += f"**Target:** {target_name} ({target_id})\n\n"
    output += f"**Disease:** {disease_name} ({disease_id})\n\n"

    if score > 0.9:
        output += "The model predicts this drug-target combination is **likely** to treat this disease."
    elif score > 0.7:
        output += "The model shows **moderate** confidence in this prediction."
    else:
        output += "The model shows **low** confidence in this prediction."

    return output


def get_stats():
    """Get model statistics."""
    eng = load_engine()
    if eng is None:
        return "Model not loaded"

    stats = eng.get_statistics()
    return f"""
**Model Statistics:**
- Drugs: {stats['num_drugs']:,}
- Targets: {stats['num_targets']:,}
- Diseases: {stats['num_diseases']:,}
- Events: {stats['num_events']:,}

**Model Performance:**
- AUC: 0.9987
- AUPR: 0.9986

**Architecture:**
- HeteroTransformerGNN
- 8-head attention
- Skip connections
"""


# Build the Gradio interface
def create_app():
    # Load engine at startup
    load_engine()

    with gr.Blocks(title="DTD-GNN Drug Repurposing", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # DTD-GNN Drug Repurposing Platform

        **Discover potential new uses for existing drugs using Graph Neural Networks**

        This platform uses a HeteroTransformerGNN model trained on drug-target-disease
        relationships to predict novel drug repurposing candidates.

        ---
        """)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tabs():
                    # Tab 1: Find Drugs for Disease
                    with gr.TabItem("Find Drugs for Disease"):
                        gr.Markdown("### Find Drug Candidates for a Disease")
                        disease_dropdown = gr.Dropdown(
                            choices=get_disease_choices(),
                            label="Select a disease",
                            value=get_disease_choices()[0] if get_disease_choices() else None
                        )
                        top_k_slider1 = gr.Slider(5, 50, value=20, step=1, label="Number of results")
                        search_btn1 = gr.Button("Find Drug Candidates", variant="primary")
                        results1 = gr.Markdown()

                        search_btn1.click(
                            find_drugs_for_disease,
                            inputs=[disease_dropdown, top_k_slider1],
                            outputs=results1
                        )

                    # Tab 2: Find Uses for Drug
                    with gr.TabItem("Find Uses for Drug"):
                        gr.Markdown("### Find Potential Uses for a Drug")
                        drug_input = gr.Textbox(
                            label="Enter Drug ID (e.g., DB00203 for Sildenafil)",
                            value="DB00203"
                        )
                        top_k_slider2 = gr.Slider(5, 50, value=20, step=1, label="Number of results")
                        search_btn2 = gr.Button("Find Disease Candidates", variant="primary")
                        results2 = gr.Markdown()

                        search_btn2.click(
                            find_diseases_for_drug,
                            inputs=[drug_input, top_k_slider2],
                            outputs=results2
                        )

                    # Tab 3: Validate Prediction
                    with gr.TabItem("Validate Prediction"):
                        gr.Markdown("### Validate a Specific Prediction")
                        gr.Markdown("Test if a specific Drug-Target-Disease combination is predicted as valid.")

                        with gr.Row():
                            val_drug = gr.Textbox(label="Drug ID", value="DB00203")
                            val_target = gr.Textbox(label="Target ID", value="P10635")
                            val_disease = gr.Textbox(label="Disease ID", value="MESH:D006973")

                        validate_btn = gr.Button("Validate Prediction", variant="primary")
                        results3 = gr.Markdown()

                        validate_btn.click(
                            validate_prediction,
                            inputs=[val_drug, val_target, val_disease],
                            outputs=results3
                        )

            # Sidebar
            with gr.Column(scale=1):
                gr.Markdown("### About")
                stats_display = gr.Markdown(get_stats())

        gr.Markdown("""
        ---
        **DTD-GNN Drug Repurposing Platform** | Built with HeteroTransformerGNN | Model AUC: 0.9987
        """)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=True)  # share=True creates public URL
