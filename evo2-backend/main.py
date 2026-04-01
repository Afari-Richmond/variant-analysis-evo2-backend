import modal
from pydantic import BaseModel

# --- Configuration & Image Build ---

evo2_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install([
        "build-essential", "cmake", "git", "wget", "unzip", "ninja-build",
        "libcudnn9-dev-cuda-12", "gcc", "g++"
    ])
    .env({
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
        "CUDA_HOME": "/usr/local/cuda",
        "CPATH": "/usr/local/cuda/include:/usr/include",
    })
    .pip_install(
        "torch==2.4.1",
        "wheel",
        "packaging",
        "ninja",
        "setuptools==69.5.1",
        "requests",
        "fastapi[standard]",  # <--- CRITICAL FIX: Required for web endpoints
        "pandas",
        "openpyxl",
        "biopython",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        extra_index_url="https://download.pytorch.org/whl/cu124"
    )
    .run_commands(
        [
            "ln -sf /usr/include/cudnn.h /usr/local/cuda/include/cudnn.h",
            "pip install flash-attn --no-build-isolation",
            "pip install --no-build-isolation 'transformer_engine[pytorch]==1.13'",
            "pip install 'vtx>=0.0.8'",
            "git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git",
            "cd evo2 && pip install --no-build-isolation -e ."
        ],
        gpu="A10G" 
    )
)

app = modal.App("variant-analysis-evo2", image=evo2_image)

volume = modal.Volume.from_name("hf_cache", create_if_missing=True)
mount_path = "/root/.cache/huggingface"

class VariantRequest(BaseModel):
    variant_position: int
    alternative: str
    genome: str
    chromosome: str

# --- Helper Functions ---

def get_genome_sequence(position, genome: str, chromosome: str, window_size=8192):
    import requests
    half_window = window_size // 2
    start = max(0, position - 1 - half_window)
    end = position - 1 + half_window + 1

    api_url = f"https://api.genome.ucsc.edu/getData/sequence?genome={genome};chrom={chromosome};start={start};end={end}"
    response = requests.get(api_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch genome sequence: {response.status_code}")

    sequence = response.json().get("dna", "").upper()
    return sequence, start

def analyze_variant(relative_pos_in_window, reference, alternative, window_seq, model):
    var_seq = window_seq[:relative_pos_in_window] + alternative + window_seq[relative_pos_in_window+1:]
    ref_score = model.score_sequences([window_seq])[0]
    var_score = model.score_sequences([var_seq])[0]
    delta_score = var_score - ref_score

    threshold = -0.0009178519
    lof_std = 0.0015140239
    func_std = 0.0009016589

    if delta_score < threshold:
        prediction = "Likely pathogenic"
        confidence = min(1.0, abs(delta_score - threshold) / lof_std)
    else:
        prediction = "Likely benign"
        confidence = min(1.0, abs(delta_score - threshold) / func_std)

    return {
        "reference": reference,
        "alternative": alternative,
        "delta_score": float(delta_score),
        "prediction": prediction,
        "classification_confidence": float(confidence)
    }

# --- Modal App Functions ---

@app.cls(gpu="H100", volumes={mount_path: volume}, max_containers=3, retries=2, scaledown_window=120)
class Evo2Model:
    @modal.enter()
    def load_evo2_model(self):
        import torch
        from _codecs import encode
        from evo2 import Evo2
        
        torch.serialization.add_safe_globals([encode])
        
        print("Loading evo2 model...")
        self.model = Evo2('evo2_7b')
        print("Evo2 model loaded")

    @modal.fastapi_endpoint(method="POST")
    def analyze_single_variant(self, request: VariantRequest):
        WINDOW_SIZE = 8192
        window_seq, seq_start = get_genome_sequence(
            position=request.variant_position,
            genome=request.genome,
            chromosome=request.chromosome,
            window_size=WINDOW_SIZE
        )

        relative_pos = request.variant_position - 1 - seq_start
        reference = window_seq[relative_pos]

        result = analyze_variant(
            relative_pos_in_window=relative_pos,
            reference=reference,
            alternative=request.alternative,
            window_seq=window_seq,
            model=self.model
        )

        result["position"] = request.variant_position
        return result

@app.function(gpu="H100", volumes={mount_path: volume}, timeout=1000)
def run_brca1_analysis():
    import base64
    import torch
    from _codecs import encode
    from io import BytesIO
    from Bio import SeqIO
    import gzip
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import roc_auc_score
    from evo2 import Evo2

    torch.serialization.add_safe_globals([encode])

    WINDOW_SIZE = 8192
    print("Loading evo2 model...")
    model = Evo2('evo2_7b')

    try:
        brca1_df = pd.read_excel('/evo2/notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx', header=2)
        with gzip.open('/evo2/notebooks/brca1/GRCh37.p13_chr17.fna.gz', "rt") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq_chr17 = str(record.seq)
                break
    except FileNotFoundError:
        return {"error": "Dataset files missing. Check git clone status."}

    brca1_df = brca1_df[['chromosome', 'position (hg19)', 'reference', 'alt', 'function.score.mean', 'func.class']]
    brca1_df.rename(columns={'chromosome': 'chrom', 'position (hg19)': 'pos', 'reference': 'ref', 'function.score.mean': 'score', 'func.class': 'class'}, inplace=True)
    brca1_df['class'] = brca1_df['class'].replace(['FUNC', 'INT'], 'FUNC/INT')

    brca1_subset = brca1_df.iloc[:500].copy()
    ref_seqs, ref_seq_to_index, ref_seq_indexes, var_seqs = [], {}, [], []

    for _, row in brca1_subset.iterrows():
        p = row["pos"] - 1
        ref_seq_start = max(0, p - WINDOW_SIZE//2)
        ref_seq_end = min(len(seq_chr17), p + WINDOW_SIZE//2)
        ref_seq = seq_chr17[ref_seq_start:ref_seq_end]
        snv_pos_in_ref = min(WINDOW_SIZE//2, p - ref_seq_start)
        var_seq = ref_seq[:snv_pos_in_ref] + row["alt"] + ref_seq[snv_pos_in_ref+1:]

        if ref_seq not in ref_seq_to_index:
            ref_seq_to_index[ref_seq] = len(ref_seqs)
            ref_seqs.append(ref_seq)
        ref_seq_indexes.append(ref_seq_to_index[ref_seq])
        var_seqs.append(var_seq)

    print("Scoring sequences...")
    ref_scores = model.score_sequences(ref_seqs)
    var_scores = model.score_sequences(var_seqs)
    delta_scores = np.array(var_scores) - np.array(ref_scores)[np.array(ref_seq_indexes)]
    brca1_subset['evo2_delta_score'] = delta_scores

    plt.figure(figsize=(4, 2))
    sns.stripplot(data=brca1_subset, x='evo2_delta_score', y='class', hue='class', palette=['#777777', 'C3'], size=2)
    plt.xlabel('Delta likelihood score, Evo 2')
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plot_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    auroc = float(roc_auc_score((brca1_subset['class'] == 'LOF'), -brca1_subset['evo2_delta_score']))
    return {"plot": plot_data, "auroc": auroc}

@app.local_entrypoint()
def main():
    print("Deployment successful. To run full analysis: modal run main.py::run_brca1_analysis")