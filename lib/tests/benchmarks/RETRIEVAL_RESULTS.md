# Retrieval Quality Benchmark

Date: 2026-08-16  |  Embedding: `openai`  |  Corpus: 3 docs, 12 chunks  |  Ground truth: 24 queries

## Overall results

| Mode | MRR | Hit@1 | Hit@3 | Hit@5 | Latency (ms) |
| --- | --- | --- | --- | --- | --- |
| `bm25` | 0.917 | 88% | 96% | 96% | 0.1 |
| `vector` | 0.979 | 96% | 100% | 100% | 194.1 |
| `hybrid` | 0.958 | 92% | 100% | 100% | 196.8 |

## By query type

### keyword

| Mode | N | MRR | Hit@1 | Hit@3 | Hit@5 |
| --- | --- | --- | --- | --- | --- |
| `bm25` | 18 | 1.000 | 100% | 100% | 100% |
| `vector` | 18 | 0.972 | 94% | 100% | 100% |
| `hybrid` | 18 | 0.972 | 94% | 100% | 100% |

### paraphrase

| Mode | N | MRR | Hit@1 | Hit@3 | Hit@5 |
| --- | --- | --- | --- | --- | --- |
| `bm25` | 6 | 0.667 | 50% | 83% | 83% |
| `vector` | 6 | 1.000 | 100% | 100% | 100% |
| `hybrid` | 6 | 0.917 | 83% | 100% | 100% |

## Per-query detail

### insurance_policy.txt

| Query | Type | Section | Rank (bm25) | Rank (vector) | Rank (hybrid) |
| --- | --- | --- | --- | --- | --- |
| employee eligibility minimum hours per week | keyword | ELIGIBILITY | #1 | #2 | #2 |
| dependent coverage age limit students | keyword | ELIGIBILITY | #1 | #1 | #1 |
| emergency room visit copay cost | keyword | COVERAGE BENEFITS | #1 | #1 | #1 |
| prescription drugs covered eighty percent deductible | keyword | COVERAGE BENEFITS | #1 | #1 | #1 |
| individual annual deductible amount fifteen hundred | keyword | DEDUCTIBLES AND OUT-OF-POCKET | #1 | #1 | #1 |
| maximum out-of-pocket limit six thousand dollars per year | keyword | DEDUCTIBLES AND OUT-OF-POCKET | #1 | #1 | #1 |
| what services are not covered by the health plan | paraphrase | EXCLUSIONS | #2 | #1 | #1 |
| free preventive checkups no out-of-pocket charge | paraphrase | COVERAGE BENEFITS | #2 | #1 | #1 |

### service_agreement.txt

| Query | Type | Section | Rank (bm25) | Rank (vector) | Rank (hybrid) |
| --- | --- | --- | --- | --- | --- |
| monthly retainer fee five thousand dollars | keyword | PAYMENT TERMS | #1 | #1 | #1 |
| invoice due date thirty days receipt | keyword | PAYMENT TERMS | #1 | #1 | #1 |
| late payment penalty two percent per month | keyword | PAYMENT TERMS | #1 | #1 | #1 |
| termination notice period sixty days | keyword | TERMINATION | #1 | #1 | #1 |
| work product intellectual property owned by client | keyword | INTELLECTUAL PROPERTY | #1 | #1 | #1 |
| governing law Delaware state agreement | keyword | GOVERNING LAW | #1 | #1 | #1 |
| immediate contract cancellation breach insolvency | paraphrase | TERMINATION | #1 | #1 | #1 |
| how disputes are resolved arbitration | paraphrase | GOVERNING LAW | #1 | #1 | #1 |

### employee_handbook.txt

| Query | Type | Section | Rank (bm25) | Rank (vector) | Rank (hybrid) |
| --- | --- | --- | --- | --- | --- |
| remote work days permitted per week | keyword | WORK HOURS AND ATTENDANCE | #1 | #1 | #1 |
| retirement contribution company match six percent salary | keyword | COMPENSATION AND BENEFITS | #1 | #1 | #1 |
| paid time off accrual rate per month | keyword | COMPENSATION AND BENEFITS | #1 | #1 | #1 |
| performance reviews conducted June December twice yearly | keyword | PERFORMANCE REVIEWS | #1 | #1 | #1 |
| harassment discrimination consequences termination | keyword | CODE OF CONDUCT | #1 | #1 | #1 |
| salary compensation annual review cycle | keyword | COMPENSATION AND BENEFITS | #1 | #1 | #1 |
| what happens after a failed performance improvement plan | paraphrase | PERFORMANCE REVIEWS | #1 | #1 | #1 |
| working from home schedule flexibility | paraphrase | WORK HOURS AND ATTENDANCE | — | #1 | #2 |

## Notes

- **BM25**: in-process keyword scoring with punctuation normalization and section-title boosting.
- **Vector**: KNN over MemoryVectorSource. With `DummyEmbeddingModel` vector similarity is random — run with `--embedding openai` for meaningful vector/hybrid results.
- **Hybrid**: Reciprocal Rank Fusion (k=60) of BM25 and vector rankings.
- chunk_size=100, overlap=20; one section body per chunk.
- Hit@k = fraction of queries where relevant chunk appears in top-k.
- MRR = mean reciprocal rank; 1.0 = always rank 1, 0.0 = never found.
