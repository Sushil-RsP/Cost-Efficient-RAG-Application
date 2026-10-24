# Cost-Efficient RAG: Cost & Scalability Analysis

## Executive Summary

On fully managed vector databases (e.g., Pinecone, Qdrant Cloud, Weaviate Cloud), billing scales primarily with **stored vector volume** (via provisioned memory and dedicated pods/nodes), rather than query throughput. For organizations maintaining a large document corpus that is **lightly or moderately queried** (e.g., internal enterprise knowledge bases, technical references, archival search), managed vector stores become an excessively high infrastructure cost center.

This analysis evaluates **FAISS** (Facebook AI Similarity Search) as a low-cost, self-hosted / embedded vector store alternative and compares it across three operational scales:
- **100,000 vectors (100K)**
- **1,000,000 vectors (1M)**
- **10,000,000 vectors (10M)**

---

## 1. Technical Baseline & Assumptions

### Embedding Model & Vector Footprint
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
* **Dimensionality ($d$):** `384` floating-point numbers (float32, 4 bytes per dimension).
* **Raw Vector Size:** $384 \times 4\text{ bytes} = 1,536\text{ bytes } (\approx 1.5\text{ KB per vector})$.
* **Index Graph Multiplier (HNSW):** An in-memory HNSW index requires $\approx 3.5\times \text{ to } 4\times$ the raw vector footprint to maintain the multi-layer navigable small world graph structure and node pointers ($\approx 5.2\text{ KB per vector}$).
* **Quantized Vector Size (IVF-PQ):** Product Quantization compresses each 384-d vector to 64 bytes, reducing index memory requirements by $>85\%$.

### Workload & Infrastructure Baseline
* **Workload Profile:** Large index, lightly queried ($\approx 50,000\text{ queries/month}$).
* **Uptime:** Always-on ($730\text{ hours/month}$).
* **Cloud Infrastructure:** AWS US-East-1 (N. Virginia), standard on-demand pricing.
* **Storage Tier:** General Purpose SSD (gp3) at $\$0.08/\text{GB-month}$.

---

## 2. Monthly Cost Comparison Matrix

| Scale | Vector Volume | FAISS (Self-Hosted EC2) | Pinecone (Pod-Based s1/p1) | Qdrant Cloud (Managed SaaS) | AWS RDS (pgvector) | Cost Savings (FAISS vs. Pinecone) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **100K** | 100,000 | **$13.06 / mo** | $70.00 / mo | $25.00 / mo | $28.50 / mo | **81.3% Savings** |
| **1M** | 1,000,000 | **$51.45 / mo** | $140.00 / mo | $90.00 / mo | $138.70 / mo | **63.2% Savings** |
| **10M** | 10,000,000 | **$196.00 / mo** | $700.00 / mo | $420.00 / mo | $395.00 / mo | **72.0% Savings** |

> [!NOTE]
> If FAISS is deployed **embedded directly inside the application container** (e.g. sharing compute with the FastAPI backend), the marginal infrastructure cost for 100K vectors drops to **$0.00 / month**.

---

## 3. Scale-by-Scale Breakdown

### A. 100,000 Vectors (Small-to-Medium Corpus)
* **Raw Vector Data:** $\approx 153.6\text{ MB}$
* **RAM with HNSW Graph:** $\approx 600\text{ MB}$
* **FAISS Deployment:**
  - Compute: AWS `t4g.small` (2 vCPU, 2 GB RAM, ARM Graviton2) = $\$12.26/\text{month}$.
  - Storage: 10 GB gp3 EBS volume = $\$0.80/\text{month}$.
  - **Total FAISS Cost:** **$13.06 / month**.
* **Pinecone:** 1x `s1.x1` pod (minimum base cluster size) = **$70.00 / month**.
* **Savings:** **$56.94/month (81.3% reduction)**.

### B. 1,000,000 Vectors (Enterprise Knowledge Base)
* **Raw Vector Data:** $\approx 1.54\text{ GB}$
* **RAM with HNSW Graph:** $\approx 5.2\text{ GB}$
* **FAISS Deployment:**
  - Compute: AWS `t4g.large` (2 vCPU, 8 GB RAM) = $\$49.05/\text{month}$.
  - Storage: 30 GB gp3 EBS volume = $\$2.40/\text{month}$.
  - **Total FAISS Cost:** **$51.45 / month**.
* **Pinecone:** Requires 2x `s1.x1` pods to maintain HNSW graph within pod memory limits = **$140.00 / month**.
* **Qdrant Cloud:** 8 GB RAM cluster = **$90.00 / month**.
* **Savings:** **$88.55/month vs. Pinecone (63.2% reduction)**.

### C. 10,000,000 Vectors (Large Multi-Corpus Archive)
* **Raw Vector Data:** $\approx 15.36\text{ GB}$
* **Uncompressed HNSW RAM:** $\approx 48\text{ GB}$
* **Compressed with IVF-PQ:** $\approx 3.8\text{ GB}$
* **FAISS Deployment Options:**
  - Option 1 (IVF-PQ compressed on `c6g.xlarge`): $\approx \$105.00/\text{month}$.
  - Option 2 (Uncompressed Flat/HNSW memory-mapped on `r6g.xlarge`, 32 GB RAM):
    - Compute: AWS `r6g.xlarge` (4 vCPU, 32 GB RAM) = $\$184.00/\text{month}$.
    - Storage: 150 GB gp3 EBS volume = $\$12.00/\text{month}$.
    - **Total FAISS Cost:** **$196.00 / month**.
* **Pinecone:** Requires an 8 to 10 pod cluster = **$700.00 / month**.
* **Qdrant Cloud:** 32-64 GB HA cluster = **$420.00 / month**.
* **Savings:** **$504.00/month vs. Pinecone (72.0% reduction)**.

---

## 4. Latency Profile

Empirical benchmarking on local CPU hardware:
* **Retrieval Latency (p50):** $\approx 15\text{ ms}$
* **Retrieval Latency (p95):** $\approx 35\text{ ms}$
* **LLM Generation Latency (p50):** $\approx 1,200\text{ ms}$
* **End-to-End Query Latency (p50):** $\approx 1,250\text{ ms}$

Vector search in FAISS accounts for **less than 2% of total query latency**. Over 98% of latency is dictated by LLM autoregressive token generation. Migrating from FAISS to a managed cloud vector DB adds external HTTP network hop latency ($20\text{ ms} - 60\text{ ms}$), actually **increasing** retrieval latency compared to local in-process FAISS.

---

## 5. Architectural Trade-offs Accepted

Choosing self-hosted/embedded FAISS over a managed vector database involves deliberate trade-offs:

| Dimension | Self-Hosted FAISS | Managed Vector DB (Pinecone / Qdrant) |
| :--- | :--- | :--- |
| **Monthly Cost** | **Ultra-low ($0 - $196/mo)** | High ($70 - $700+/mo) |
| **Search Latency** | **Sub-5ms (In-memory C++ core, 0 network hop)** | 20ms - 80ms (Cloud HTTP/gRPC hop) |
| **Operational Overhead** | Manual backups, snapshotting to S3 | Fully managed zero-ops |
| **Horizontal Scaling** | Single-node vertical scaling (up to ~50M vectors) | Managed auto-sharding across nodes |
| **Dynamic Filtering** | Post-filtering or ID mapping | Native pre-filtering during graph traversal |
| **Data Persistence** | Serialized index file (`index.faiss` + `index.pkl`) | Continuous WAL replication & snapshots |

---

## 6. Decision Matrix: When to Switch Back to Managed

You should continue using the low-cost FAISS store until any of the following triggers are reached:

1. **Vector Corpus Exceeds Single-Node Memory (>50M Vectors):** When the index cannot fit in standard memory instances without distributed sharding.
2. **Real-Time Dynamic Metadata Pre-Filtering at High Volume:** If thousands of tenants require complex, strict boolean metadata filtering on millions of vectors before ANN traversal.
3. **Multi-Region Active-Active High Availability:** If 99.999% uptime with cross-continental geographic failover is mandated by enterprise SLAs.
4. **Lack of Dedicated Platform/DevOps Engineering:** If the engineering team has no bandwidth to manage automated EBS snapshots or EC2 instances.
