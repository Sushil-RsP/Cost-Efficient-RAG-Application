# Comprehensive Natural Language Processing (NLP) Technical Guide

A structured reference covering text processing pipelines, numerical representation models, sequence architectures, modern Transformer models, Large Language Models (LLMs), and evaluation frameworks.

---

## 1. Text Preprocessing Pipeline & Tokenization

### A. Core Preprocessing Steps
* **Normalization:** Lowercasing, Unicode NFKC normalization, removing accent marks and noise.
* **Stop-Word Removal:** Eliminating high-frequency structural words (e.g., *the*, *is*, *in*) when building statistical representations (TF-IDF, Bag-of-Words).
* **Lemmatization vs. Stemming:**
  * **Stemming (e.g., Porter Stemmer):** Heuristic rule-based truncation. Fast, but produces non-dictionary tokens (e.g., *"running"* $\rightarrow$ *"runn"*).
  * **Lemmatization (e.g., WordNet):** Morphological analysis mapped to actual dictionary canonical forms (lemmas) using Part-of-Speech tags (e.g., *"better"* $\rightarrow$ *"good"*).

### B. Modern Tokenization Algorithms

| Algorithm | Mechanism | Primary Use Case |
| :--- | :--- | :--- |
| **Byte-Pair Encoding (BPE)** | Iteratively merges the most frequent pairs of adjacent characters or subwords. | GPT series, RoBERTa, LLaMA |
| **WordPiece** | Iteratively merges pairs that maximize the language model likelihood of the training data. | BERT, DistilBERT |
| **Unigram** | Starts with a large vocabulary and iteratively drops subwords that minimize entropy loss. | SentencePiece, T5, ALBERT |

---

## 2. Text Representation & Vector Embeddings

### A. Term Frequency-Inverse Document Frequency (TF-IDF)
Measures how informative a term $t$ is relative to a document $d$ within a corpus $D$.

$$
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
$$

Where:

$$
\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}, \quad \text{IDF}(t, D) = \ln\left(\frac{|D|}{1 + |\{d \in D : t \in d\}|}\right)
$$

### B. Dense Word Embeddings
1. **Word2Vec (Skip-Gram & CBOW):** Learns dense static vectors ($d \in [100, 300]$) using shallow neural network prediction objectives.
2. **GloVe (Global Vectors):** Factorizes global word-word co-occurrence matrix log-probabilities.
3. **Limitations of Static Embeddings:** Polysemy problem—the word *"bank"* receives the exact same vector representation in *"river bank"* and *"bank account"*.

---

## 3. Sequence Models & Attention Mechanism

### A. Recurrent Neural Networks (RNNs) & LSTMs
Traditional sequential architectures process text step-by-step ($t_1, t_2, \dots, t_n$), introducing vanishing/exploding gradients over long contexts.

* **Long Short-Term Memory (LSTM):** Controls information flow via three gates:
  * **Input Gate:** $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$
  * **Forget Gate:** $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$
  * **Output Gate:** $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$

### B. Scaled Dot-Product Attention
Computes dynamic alignment weights across sequence elements, enabling parallel processing.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where $Q$ (Query), $K$ (Key), and $V$ (Value) are projections of input embeddings, and $d_k$ represents the key dimension size scaling factor.

---

## 4. Transformer Architectures & LLM Foundations

```
    Encoder-Only                   Decoder-Only                 Encoder-Decoder
  (e.g., BERT, RoBERTa)          (e.g., GPT-4, LLaMA)            (e.g., T5, BART)
         │                              │                              │
  Bidirectional Context           Causal / Autoregressive         Sequence-to-Sequence
  Classification, NER, Search     Generative Text, Code          Translation, Summarization
```

### A. Architecture Classification
* **Encoder-Only (Autoencoding):** Pre-trained via Masked Language Modeling (MLM). Attends to context bi-directionally. Best for classification, extraction, and embedding generation.
* **Decoder-Only (Autoregressive):** Pre-trained via Causal Language Modeling (CLM) predicting $P(w_t \mid w_1, \dots, w_{t-1})$. Uses causal masking to prevent looking ahead. Best for generation.
* **Encoder-Decoder (Seq2Seq):** Encodes full input context bi-directionally and generates output tokens auto-regressively. Best for machine translation and abstractive summarization.

---

## 5. Vector Search & Retrieval-Augmented Generation (RAG)

### A. Similarity Metrics

* **Cosine Similarity:**
  

  $$
  \text{Cosine Sim}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}
  $$

* **Euclidean Distance ($L_2$):** $\|A - B\|_2 = \sqrt{\sum_{i=1}^n (A_i - B_i)^2}$

### B. Vector Indexing Strategies
1. **HNSW (Hierarchical Navigable Small World):** Multi-layer graph index balancing search speed and recall accuracy.
2. **IVF (Inverted File Index):** Clusters vector space into Voronoi cells to limit nearest-neighbor search scope.

---

## 6. Evaluation Frameworks

### A. Natural Language Generation (NLG) Metrics
* **BLEU (Bilingual Evaluation Understudy):** Precision-focused $n$-gram overlap metric with brevity penalty for translation.
* **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** Recall-focused metric for summarization.
  * **ROUGE-N:** $n$-gram co-occurrence recall.
  * **ROUGE-L:** Longest Common Subsequence (LCS) score.

### B. Classification & Retrieval Metrics
* **Mean Reciprocal Rank (MRR):** Evaluates rank position of first correct retrieved item:
  

  $$
  \text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
  $$

* **Normalized Discounted Cumulative Gain (NDCG@K):** Measures ranked retrieval quality with non-binary relevance grading.

---

## 7. Python Implementation: Transformer Pipeline

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Load Pre-trained Encoder Model for Semantic Embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def mean_pooling(model_output, attention_mask):
    """Applies attention-mask-weighted mean pooling to extract sequence vectors."""
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Sample Documents
sentences = [
    "The data analyst built a machine learning dashboard.",
    "A predictive model was deployed by the analytics team."
]

# Tokenize and encode
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Model Inference
with torch.no_grad():
    model_output = model(**encoded_input)

# Compute Embeddings and Calculate Cosine Similarity
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

similarity = torch.mm(sentence_embeddings[0:1], sentence_embeddings[1:2].T)
print(f"Cosine Similarity: {similarity.item():.4f}")
```

---

## 8. Production Checklist for NLP Deployment

* [ ] Subword tokenizer handles out-of-vocabulary (OOV) edge cases cleanly.
* [ ] Max sequence truncation length set according to dataset percentile coverage.
* [ ] Quantization applied (`FP16`, `INT8`, or `INT4`) to optimize inference speed and VRAM footprint.
* [ ] Context window limits and retrieval chunk sizes balanced for RAG pipelines.
* [ ] Guardrails configured for toxicity, prompt injection, and PII detection.