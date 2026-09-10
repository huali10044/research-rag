# RAG Evaluation Report

Generated: 2026-09-07T22:52:38

## Retrieval and generation quality (content + metadata queries)

| Config | n | faithfulness | answer_relevancy | llm_context_precision_with_reference | context_recall |
|---|---|---|---|---|---|
| top_k=5 | 18 | 0.864 | 0.691 | 0.825 | 0.972 |

## Deterministic checks (stats + listing queries)

| Config | n | pass rate | query-type accuracy |
|---|---|---|---|
| top_k=5 | 8 | 100% | 100% |

## Routing accuracy (classify_query, all query types)

| Config | overall | stats | listing | metadata | content |
|---|---|---|---|---|---|
| top_k=5 | 96% | 100% | 100% | 75% | 100% |

## Routing mismatches

- `meta-02` (top_k=5): expected `metadata`, got `content`
