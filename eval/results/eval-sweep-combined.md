# RAG Evaluation Report

Generated: 2026-09-13T22:40:37

## Retrieval and generation quality (content + metadata queries)

| Config | n | faithfulness | answer_relevancy | llm_context_precision_with_reference | context_recall |
|---|---|---|---|---|---|
| top_k=3 | 18 | 0.650 | 0.802 | 0.632 | 0.528 |
| top_k=5 (run 1) | 18 | 0.609 | 0.858 | 0.597 | 0.583 |
| top_k=5 (run 2) | 18 | 0.554 | 0.864 | 0.658 | 0.625 |
| top_k=8 (run 1) | 18 | 0.767 | 0.900 | 0.655 | 0.833 |
| top_k=8 (run 2) | 18 | 0.590 | 0.804 | 0.585 | 0.611 |
| top_k=8 (run 3) | 18 | 0.640 | 0.835 | 0.687 | 0.625 |
| top_k=12 | 18 | 0.657 | 0.827 | 0.619 | 0.639 |

## Deterministic checks (stats + listing queries)

| Config | n | pass rate | query-type accuracy |
|---|---|---|---|
| top_k=3 | 8 | 100% | 100% |
| top_k=5 (run 1) | 8 | 100% | 100% |
| top_k=5 (run 2) | 8 | 100% | 100% |
| top_k=8 (run 1) | 8 | 100% | 100% |
| top_k=8 (run 2) | 8 | 100% | 100% |
| top_k=8 (run 3) | 8 | 100% | 100% |
| top_k=12 | 8 | 100% | 100% |

## Routing accuracy (classify_query, all query types)

| Config | overall | stats | listing | metadata | content |
|---|---|---|---|---|---|
| top_k=3 | 100% | 100% | 100% | 100% | 100% |
| top_k=5 (run 1) | 100% | 100% | 100% | 100% | 100% |
| top_k=5 (run 2) | 100% | 100% | 100% | 100% | 100% |
| top_k=8 (run 1) | 100% | 100% | 100% | 100% | 100% |
| top_k=8 (run 2) | 100% | 100% | 100% | 100% | 100% |
| top_k=8 (run 3) | 100% | 100% | 100% | 100% | 100% |
| top_k=12 | 100% | 100% | 100% | 100% | 100% |
