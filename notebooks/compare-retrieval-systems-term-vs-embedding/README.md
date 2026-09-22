# Comparative Retrieval Systems: Term-based (BM25) vs. Dense Embedding vs. Hybrid (RRF)

Empirical Information Retrieval evaluation comparing lexical search, dense bi-encoder models, and hybrid rank fusion directly against two canonical academic benchmarks from the **BEIR (Benchmarking IR)** suite:

1. **`SciFact` (5,183 passages, 300 test queries)**: Scientific claim verification with exact biological, genetic, and chemical nomenclature (**high lexical bias**).
2. **`NFCorpus` (3,633 passages, 323 test queries)**: NutritionFacts medical QA pairing colloquial patient questions with technical PubMed abstracts (**high semantic/vocabulary mismatch**).

---

## Retrieval Approaches Compared

1. **BM25Okapi** (Exact lexical matching with $TF$-$IDF$ and document length normalization)
2. **Dense: `all-MiniLM-L6-v2`** (384-d symmetric general-purpose bi-encoder)
3. **Dense: `BAAI/bge-small-en-v1.5`** (384-d asymmetric bi-encoder with query instruction prefix)
4. **Hybrid: Reciprocal Rank Fusion (RRF)** ($k=60$)

---

## Setup & Reproduction

### 1. Requirements

Install dependencies using the virtual environment:

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Or using standard pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launching the Notebook

```bash
source .venv/bin/activate
jupyter notebook retrieval_comparison.ipynb
```

Click **Kernel ➔ Restart & Run All**.

---

## Execution Speed & Acceleration
- **GPU Accelerated**: If an NVIDIA GPU is present (e.g., RTX 3050+), pre-encoding 5,000+ passages takes only **~1.5 seconds per model**.
- **CPU Fallback**: Automatically falls back to CPU if no GPU is found.
- Total notebook run time across both datasets is **~30 to 45 seconds**.
