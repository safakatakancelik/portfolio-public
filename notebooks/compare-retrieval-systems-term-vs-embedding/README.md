# IR Retrieval Evaluation: Term-Based Sparse vs. Embedding-Based Dense

### Retrieval Models Compared
- **BM25 (Sparse)**: Inverted index matching using TF-IDF weighting and document length normalization.
- **all-MiniLM-L6-v2 (Dense)**: 384-dimensional symmetric bi-encoder; maps text to continuous latent semantic vectors.
- **BGE-small-en-v1.5 (Dense + Instruction)**: 384-dimensional asymmetric bi-encoder; uses a task instruction prefix on queries to align search intent.
- **Hybrid RRF (k=60)**: Reciprocal Rank Fusion ($Score = \sum \frac{1}{60 + rank}$); merges term-based sparse and embedding-based dense ranks.

### The Contrastive Benchmarks
- **BEIR SciFact** (5,183 docs, 300 queries): Scientific claim verification. Features exact gene acronyms, chemical compounds, and precise technical jargon (**High Term-Based Sparse Bias**).
- **BEIR NFCorpus** (3,633 docs, 323 queries): Nutrition/medical QA. Pairs everyday, conversational health queries with technical PubMed abstracts (**High Semantic / Vocabulary Mismatch favoring Embedding-Based Dense**).

### Core Questions & Learning Points
1. **How does embedding-based dense models perform against exact word matching on domain-specific technical terms?**  
   - In such cases term-based sparse models have potential to outperform complex dense models e.g. BM25 on SciFact against MiniLM failing to domain shift.
2. **When does embedding-based dense models show superiority compared to lexical models?**
   - When queries rely on semantics using paraphrasing, or layman terms as in NFCorpus. Term-based sparse models collapse due to vocabulary mismatch, while embedding-based dense models bridge the synonym gap.
3. **What is the benefit of Hybrid RRF models?**  
   - Protects against single-model catastrophic failure. If the dense model hallucinates or BM25 suffers a vocabulary mismatch, the other system keeps the relevant document in the top results.  
4. **What to do in the real-world?:** System choice is a trade-off between **infrastructure cost and latency** (term-based sparse models run on cheap CPUs with sub-millisecond lookups) versus **semantic recall** (embedding-based dense models require GPU embedding pipelines and vector DB storage); best to experiment and make smart decisions.
---

## Metric Cheat Sheet
- **Hit@k**: Did at least one relevant doc appear in top-$k$? (Binary: 0 or 1). Best for RAG context verification.
- **MRR@k**: How fast was the *first* hit found? ($\frac{1}{\text{rank}}$). Best for single-answer / QA search.
- **nDCG@k**: Did the *best* docs rank at the top? Uses graded relevance with logarithmic position decay. The gold standard for multi-result search.

---

## Benchmark Results Summary

Evaluated over 100 test queries each on **BEIR SciFact** (scientific claims) and **BEIR NFCorpus** (medical QA):

| Retriever | SciFact Hit@1 | SciFact MRR@10 | SciFact nDCG@10 | NFCorpus Hit@1 | NFCorpus MRR@10 | NFCorpus nDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25** (Term-Based Sparse) | 0.5900 | 0.6979 | 0.7290 | 0.4600 | 0.5436 | 0.3396 |
| **all-MiniLM-L6-v2** (Embedding-Based Dense) | 0.5700 | 0.6673 | 0.6996 | 0.4800 | 0.5611 | 0.3530 |
| **BGE-small-en-v1.5** (Dense + Instruction) | 0.6500 | 0.7182 | 0.7421 | **0.5400** | **0.6139** | **0.3987** |
| **Hybrid (BM25 + BGE RRF)** | **0.7100** | **0.7672** | **0.7852** | **0.5400** | 0.6069 | 0.3762 |

---

## Model Comparison & Conclusions

| Approach | Strengths | Failure Mode | Best Used When |
| :--- | :--- | :--- | :--- |
| **BM25** (Term-Based Sparse) | Instant indexing (<1s), zero training, flawless on exact codes, SKUs & names. | Total failure on synonyms and paraphrasing (vocabulary mismatch). | Keywords, catalog numbers, entity-heavy domains. |
| **MiniLM-L6** (Embedding-Based Dense) | Fast, lightweight 384-d bi-encoder. Good on general web QA. | Struggles on rare, technical, out-of-distribution domain terms. | General English search on a strict CPU/memory budget. |
| **BGE-small** (Embedding-Based Dense) | Instruction prefix aligns query intent; beats BM25 across domains. | Slower to encode than BM25; can still dilute exact IDs. | Asymmetric query-to-passage search and semantic Q&A. |
| **Hybrid RRF** (Sparse + Dense) | **Clear overall winner.** Combines keyword precision + latent semantics. | Requires maintaining two indices and subject to pool poisoning. | **Default for production RAG and enterprise search.** |
