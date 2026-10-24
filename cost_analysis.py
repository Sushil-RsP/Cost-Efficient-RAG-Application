import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Assumptions:
# Vector dimensionality: 384 (all-MiniLM-L6-v2)
# Raw float32 vector size: 384 * 4 bytes = 1,536 bytes (1.5 KB)
# HNSW index multiplier: ~3.5x - 4x for graph edges and vector cache
# Monthly query volume: 50,000 queries (lightly queried workload)
# Cloud Provider baseline: AWS US-East (N. Virginia), on-demand pricing, 730 hours/month

SCALES = [
    {
        "vector_count": 100_000,
        "scale_label": "100K Vectors",
        "raw_data_mb": 153.6,
        "hnsw_index_ram_gb": 0.6,
        "faiss_tier": {
            "instance": "AWS t4g.small (2 vCPU, 2 GB RAM)",
            "compute_cost": 12.26,
            "ebs_cost": 0.80, # 10 GB gp3
            "total_monthly": 13.06,
            "note": "Fits easily in 2 GB RAM. Can also run embedded ($0 additional)."
        },
        "pinecone_pod": {
            "tier": "1x s1.x1 pod (always-on)",
            "monthly_cost": 70.00,
            "note": "Base tier minimum requirement for pod-based index."
        },
        "qdrant_cloud": {
            "tier": "1 GB RAM Cluster (Single Node)",
            "monthly_cost": 25.00,
            "note": "Managed SaaS cluster minimum baseline."
        },
        "pgvector_rds": {
            "tier": "AWS db.t4g.micro / small (2 GB RAM)",
            "monthly_cost": 28.50,
            "note": "Managed PostgreSQL with pgvector extension."
        }
    },
    {
        "vector_count": 1_000_000,
        "scale_label": "1M Vectors",
        "raw_data_mb": 1536.0,
        "hnsw_index_ram_gb": 5.2,
        "faiss_tier": {
            "instance": "AWS t4g.large (2 vCPU, 8 GB RAM)",
            "compute_cost": 49.05,
            "ebs_cost": 2.40, # 30 GB gp3
            "total_monthly": 51.45,
            "note": "Full in-memory HNSW index with sub-5ms query latency."
        },
        "pinecone_pod": {
            "tier": "2x s1.x1 or 1x p1.x2 pods",
            "monthly_cost": 140.00,
            "note": "Requires scaling pods to maintain index in RAM."
        },
        "qdrant_cloud": {
            "tier": "8 GB RAM Managed Cluster",
            "monthly_cost": 90.00,
            "note": "Standard managed Qdrant cloud instance."
        },
        "pgvector_rds": {
            "tier": "AWS db.m6g.large (8 GB RAM)",
            "monthly_cost": 138.70,
            "note": "Buffer pool must hold HNSW graph in memory."
        }
    },
    {
        "vector_count": 10_000_000,
        "scale_label": "10M Vectors",
        "raw_data_mb": 15360.0,
        "hnsw_index_ram_gb": 48.0,
        "faiss_tier": {
            "instance": "AWS r6g.xlarge (4 vCPU, 32 GB RAM) with IVF-PQ / Memmap",
            "compute_cost": 184.00,
            "ebs_cost": 12.00, # 150 GB gp3
            "total_monthly": 196.00,
            "note": "Uses FAISS IVF-PQ (64 bytes/vector) consuming <4 GB RAM, or uncompressed on r6g."
        },
        "pinecone_pod": {
            "tier": "8x to 10x s1.x1 pods",
            "monthly_cost": 700.00,
            "note": "High cost for lightly-queried enterprise index."
        },
        "qdrant_cloud": {
            "tier": "32-64 GB RAM HA Multi-node Cluster",
            "monthly_cost": 420.00,
            "note": "Enterprise multi-replica cluster."
        },
        "pgvector_rds": {
            "tier": "AWS db.r6g.xlarge (32 GB RAM, Multi-AZ)",
            "monthly_cost": 395.00,
            "note": "Dedicated database instance with read replica."
        }
    }
]


def generate_cost_report():
    print("=" * 80)
    print("COST-EFFICIENT RAG: VECTOR STORE COST COMPARISON TABLE (MONTHLY)")
    print("=" * 80)

    rows = []
    for scale in SCALES:
        faiss = scale["faiss_tier"]["total_monthly"]
        pinecone = scale["pinecone_pod"]["monthly_cost"]
        qdrant = scale["qdrant_cloud"]["monthly_cost"]
        pgvector = scale["pgvector_rds"]["monthly_cost"]

        savings_vs_pinecone = ((pinecone - faiss) / pinecone) * 100
        savings_vs_qdrant = ((qdrant - faiss) / qdrant) * 100

        rows.append({
            "Scale": scale["scale_label"],
            "Vectors": f"{scale['vector_count']:,}",
            "FAISS (Self-Hosted/Embedded)": f"${faiss:.2f}/mo",
            "Pinecone (Pod-based)": f"${pinecone:.2f}/mo",
            "Qdrant Cloud": f"${qdrant:.2f}/mo",
            "pgvector (AWS RDS)": f"${pgvector:.2f}/mo",
            "Monthly Savings vs Pinecone": f"{savings_vs_pinecone:.1f}%",
            "Monthly Savings vs Qdrant": f"{savings_vs_qdrant:.1f}%"
        })

    # Output JSON table
    out_json = BASE_DIR / "cost_comparison_table.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "assumptions": {
                "embedding_model": "all-MiniLM-L6-v2 (384 dimensions)",
                "raw_bytes_per_vector": 1536,
                "workload_profile": "Large index, lightly-queried (50,000 queries/month)",
                "uptime": "730 hours/month (always-on)",
                "compute_pricing_basis": "AWS US-East-1 on-demand 2026 pricing"
            },
            "comparison_matrix": rows,
            "detailed_scales": SCALES
        }, f, indent=2)

    print(f"\nSaved cost matrix to: {out_json}")
    return rows


if __name__ == "__main__":
    generate_cost_report()
