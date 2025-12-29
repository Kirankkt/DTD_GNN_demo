"""
DTD-GNN Drug Repurposing Platform

A web interface for discovering potential drug repurposing candidates
using the HeteroTransformerGNN model.

Run locally:
    streamlit run app.py

Deploy on Streamlit Cloud or HuggingFace Spaces for free!
"""
import streamlit as st
import torch
import pandas as pd
import os

# Page config
st.set_page_config(
    page_title="DTD-GNN Drug Repurposing",
    page_icon="💊",
    layout="wide"
)

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
    # Additional common drugs
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


@st.cache_resource
def load_engine():
    """Load the discovery engine (cached)."""
    from engine.hetero_discovery import HeteroDiscoveryEngine

    model_path = "data/dtd_full_checkpoint.pt"
    data_path = "data/dtd_full_graph.pt"

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        return None

    return HeteroDiscoveryEngine(model_path, data_path)


def get_name(id_val, name_dict, prefix=""):
    """Get human-readable name or return ID."""
    return name_dict.get(id_val, f"{prefix}{id_val}")


def main():
    # Header
    st.title("💊 DTD-GNN Drug Repurposing Platform")
    st.markdown("""
    **Discover potential new uses for existing drugs using Graph Neural Networks**

    This platform uses a HeteroTransformerGNN model trained on drug-target-disease
    relationships to predict novel drug repurposing candidates.

    ---
    """)

    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("""
    **Model Performance:**
    - AUC: 0.9987
    - AUPR: 0.9986
    - Trained on 300K events

    **Architecture:**
    - HeteroTransformerGNN
    - 8-head attention
    - Skip connections

    **Data:** BioSNAP dataset
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")

    # Load engine
    engine = load_engine()

    if engine is None:
        st.error("""
        ⚠️ Model not found! Please train the model first:
        ```
        python train_scalable.py --events 300000 --epochs 100
        ```
        """)
        return

    # Display stats in sidebar
    stats = engine.get_statistics()
    st.sidebar.metric("Drugs", f"{stats['num_drugs']:,}")
    st.sidebar.metric("Targets", f"{stats['num_targets']:,}")
    st.sidebar.metric("Diseases", f"{stats['num_diseases']:,}")
    st.sidebar.metric("Events", f"{stats['num_events']:,}")

    # Main content - tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Find Drugs for Disease", "💊 Find Uses for Drug", "📊 Validate Prediction"])

    # Tab 1: Find drugs for disease
    with tab1:
        st.header("Find Drug Candidates for a Disease")

        # Build disease options
        available_diseases = []
        for mesh_id in engine.disease_to_id.keys():
            name = DISEASE_NAMES.get(mesh_id, mesh_id)
            available_diseases.append((name, mesh_id))

        # Sort by name
        available_diseases.sort(key=lambda x: x[0])

        # Prioritize known diseases at the top
        known_first = [d for d in available_diseases if not d[0].startswith("MESH:")]
        unknown = [d for d in available_diseases if d[0].startswith("MESH:")]
        available_diseases = known_first + unknown

        # Disease selector
        disease_options = [f"{name} ({mesh_id})" for name, mesh_id in available_diseases]
        selected_disease = st.selectbox(
            "Select a disease:",
            disease_options,
            index=0
        )

        # Parse selection
        mesh_id = selected_disease.split("(")[-1].rstrip(")")

        col1, col2 = st.columns([1, 3])
        with col1:
            top_k = st.slider("Number of results:", 5, 50, 20)

        if st.button("🔍 Find Drug Candidates", type="primary"):
            with st.spinner("Searching..."):
                results = engine.find_drugs_for_disease(mesh_id, top_k=top_k)

            if results:
                st.success(f"Found {len(results)} potential drug candidates!")

                # Format results
                df_data = []
                for r in results:
                    drug_name = get_name(r['drug'], DRUG_NAMES)
                    target_name = get_name(r['target'], TARGET_NAMES)

                    df_data.append({
                        "Rank": len(df_data) + 1,
                        "Drug": drug_name,
                        "Drug ID": r['drug'],
                        "Target": target_name,
                        "Target ID": r['target'],
                        "Score": f"{r['score']:.4f}",
                        "Confidence": r['confidence']
                    })

                df = pd.DataFrame(df_data)

                # Style the dataframe
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    f"drug_candidates_{mesh_id}.csv",
                    "text/csv"
                )

                # Lookup helper
                st.markdown("---")
                st.markdown("### 🔍 Look Up IDs")
                st.markdown("""
                **Click to look up any ID:**
                - **Drug IDs (DB...)**: [DrugBank](https://go.drugbank.com/drugs/) - add ID to URL (e.g., `go.drugbank.com/drugs/DB00721`)
                - **Target IDs (P.../Q...)**: [UniProt](https://www.uniprot.org/uniprotkb/) - add ID to URL (e.g., `uniprot.org/uniprotkb/P35354`)
                - **Disease IDs (MESH:D...)**: [MeSH](https://meshb.nlm.nih.gov/record/ui?ui=) - add ID after `ui=` (e.g., `ui=D001943`)
                """)
            else:
                st.warning("No results found.")

    # Tab 2: Find uses for drug
    with tab2:
        st.header("Find Potential Uses for a Drug")

        # Drug search
        drug_query = st.text_input("Enter Drug ID (e.g., DB00203 for Sildenafil):", "DB00203")

        if st.button("🔍 Find Disease Candidates", type="primary"):
            if drug_query:
                # Search for drug
                matches = engine.search_drugs(drug_query)

                if matches:
                    with st.spinner("Searching..."):
                        results = engine.find_diseases_for_drug(matches[0], top_k=20)

                    if results:
                        drug_name = get_name(matches[0], DRUG_NAMES)
                        st.success(f"Found {len(results)} potential uses for {drug_name}!")

                        df_data = []
                        for r in results:
                            disease_name = get_name(r['disease'], DISEASE_NAMES)
                            target_name = get_name(r['target'], TARGET_NAMES)

                            df_data.append({
                                "Disease": disease_name,
                                "Disease ID": r['disease'],
                                "Target": target_name,
                                "Score": f"{r['score']:.4f}"
                            })

                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No results found for this drug.")
                else:
                    st.error(f"Drug '{drug_query}' not found in database.")

    # Tab 3: Validate prediction
    with tab3:
        st.header("Validate a Specific Prediction")

        st.markdown("Test if a specific Drug-Target-Disease combination is predicted as valid.")

        col1, col2, col3 = st.columns(3)

        with col1:
            drug_id = st.text_input("Drug ID:", "DB00203")
        with col2:
            target_id = st.text_input("Target ID:", "P10635")
        with col3:
            disease_id = st.text_input("Disease ID:", "MESH:D006973")

        if st.button("✅ Validate Prediction", type="primary"):
            result = engine.predict_triplet(drug_id, target_id, disease_id)

            if result['valid']:
                score = result['score']

                # Color based on score
                if score > 0.9:
                    st.success(f"""
                    ### ✅ High Confidence Prediction

                    **Score:** {score:.4f}

                    The model predicts this drug-target combination is **likely** to treat this disease.
                    """)
                elif score > 0.7:
                    st.warning(f"""
                    ### ⚠️ Medium Confidence Prediction

                    **Score:** {score:.4f}

                    The model shows **moderate** confidence in this prediction.
                    """)
                else:
                    st.info(f"""
                    ### ℹ️ Low Confidence Prediction

                    **Score:** {score:.4f}

                    The model shows **low** confidence in this prediction.
                    """)
            else:
                st.error(f"Could not validate: {result.get('message', 'Unknown error')}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>DTD-GNN Drug Repurposing Platform | Built with HeteroTransformerGNN</p>
        <p>Model AUC: 0.9987 | Based on BioSNAP dataset</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
