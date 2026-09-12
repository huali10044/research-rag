# RAG Evaluation Report

Generated: 2026-09-12T02:21:50

## Retrieval and generation quality (content + metadata queries)

| Config | n | faithfulness | answer_relevancy | llm_context_precision_with_reference | context_recall |
|---|---|---|---|---|---|
| top_k=3 | 18 | 0.650 | 0.802 | 0.632 | 0.528 |
| top_k=5 | 18 | 0.609 | 0.858 | 0.597 | 0.583 |
| top_k=8 | 18 | 0.767 | 0.900 | 0.655 | 0.833 |

## Deterministic checks (stats + listing queries)

| Config | n | pass rate | query-type accuracy |
|---|---|---|---|
| top_k=3 | 8 | 100% | 100% |
| top_k=5 | 8 | 100% | 100% |
| top_k=8 | 8 | 100% | 100% |

## Routing accuracy (classify_query, all query types)

| Config | overall | stats | listing | metadata | content |
|---|---|---|---|---|---|
| top_k=3 | 100% | 100% | 100% | 100% | 100% |
| top_k=5 | 100% | 100% | 100% | 100% | 100% |
| top_k=8 | 100% | 100% | 100% | 100% | 100% |
