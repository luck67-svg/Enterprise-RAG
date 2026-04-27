# Hybrid Retriever Design

**Date:** 2026-04-27

**Status:** Approved in chat, pending written spec review

## Goal

Replace the current dense-only child-chunk retrieval path with an application-layer hybrid retriever that combines:

- dense retrieval from Qdrant using `bge-m3`
- lexical BM25 retrieval over local child chunks
- Reciprocal Rank Fusion (RRF) to merge both result lists
- existing parent expansion and cross-encoder reranking

The change must preserve the current parent-child chunking model and avoid rebuilding the existing Qdrant collection.

## Current State

The retrieval pipeline is currently implemented in [app/rag/chain.py](/D:/Enterprise-RAG/app/rag/chain.py) and behaves as:

1. retrieve child chunks from Qdrant with dense cosine similarity
2. expand matched child chunks into deduplicated parent chunks
3. rerank parent chunks through `app/rag/reranker.py`
4. return the top `retrieval_top_k` parent chunks to the prompt

The project already depends on `rank-bm25`, but there is no lexical retrieval or fusion layer in the current code.

## Non-Goals

- no Qdrant schema migration
- no Qdrant sparse vectors or native hybrid retrieval
- no replacement of the reranker service
- no unrelated refactor of upload, loading, or generation logic
- no incremental inverted-index maintenance in the first version

## Proposed Architecture

### Retrieval Flow

The new retrieval flow will be:

1. dense child retrieval from Qdrant
2. BM25 child retrieval from a local chunk corpus
3. RRF fusion on child chunk ids
4. parent expansion from fused child hits
5. parent deduplication and score aggregation
6. existing reranker over parent chunks
7. final top-k selection

### Modules

#### `app/rag/bm25_index.py`

Owns the local BM25 corpus and query path.

Responsibilities:

- store child chunk records needed for lexical retrieval
- rebuild an in-memory `BM25Okapi` index from persisted records
- expose add, delete, load, rebuild, and search operations
- provide a stable tokenization function for both indexing and querying
- degrade cleanly when the corpus is missing or empty

The first version will persist raw chunk records and rebuild the in-memory BM25 index after each document mutation. It will not try to mutate the BM25 structure in place.

#### `app/rag/hybrid_retriever.py`

Owns hybrid retrieval orchestration.

Responsibilities:

- run dense child retrieval
- run BM25 child retrieval
- fuse rankings with RRF
- deduplicate by `chunk_id`
- expand to parents
- aggregate child fused scores into parent ordering
- invoke the existing reranker on parent documents
- apply final top-k truncation

This keeps `app/rag/chain.py` focused on chain assembly instead of retrieval internals.

#### `app/rag/splitter.py`

Keeps parent-child chunking logic but must emit a stable `chunk_id` for every child chunk. Hybrid fusion depends on a shared identifier between dense results and BM25 corpus records.

Each child chunk should also retain:

- `source`
- `page`
- `parent_id`
- `parent_content`

#### `app/api/kb.py`

Must update both storage systems during document mutations:

- upload: write dense chunks to Qdrant and child chunk records to BM25 storage
- replace existing file: delete prior Qdrant points and prior BM25 chunk records
- delete document: remove Qdrant points and BM25 chunk records

#### `app/rag/chain.py`

Must stop constructing retrieval logic inline and instead depend on a single hybrid retriever entry point.

## Data Model

The BM25 persisted corpus should be a readable file, for example:

- `data/bm25_chunks.json`

Each record should represent one child chunk and include:

- `chunk_id`
- `source`
- `page`
- `parent_id`
- `parent_content`
- `page_content`

This file is the source of truth for rebuilding the in-memory BM25 index on startup.

## Fusion Strategy

### Why Fuse at Child-Chunk Level

The existing system retrieves child chunks and only expands to parents after retrieval. BM25 should use the same unit of retrieval so that:

- dense and lexical rankings are directly comparable
- RRF can merge equivalent hits by `chunk_id`
- parent expansion semantics remain unchanged

### RRF Formula

For each ranked list:

`score += 1 / (rrf_k + rank)`

Where:

- `rank` is one-based rank within the source list
- `rrf_k` is configurable

RRF is preferred here because dense similarity scores and BM25 scores are not directly comparable. Rank-based fusion avoids brittle score normalization.

### Parent Ordering Before Rerank

After fusion at child level, parent documents should be deduplicated before reranking.

Recommended first-version rule:

- collect fused children in descending fused-score order
- when multiple children map to the same parent, aggregate parent score by summing fused scores
- preserve one parent document per `parent_id`
- sort parents by aggregated fused score before sending to reranker

This gives the reranker a better candidate order and preserves signals from multiple relevant children.

## Persistence and Rebuild Strategy

The BM25 Python object should not be serialized directly.

Instead:

1. persist chunk records as JSON
2. on startup, load records and rebuild tokenized corpus in memory
3. on upload or delete, update records and rebuild the in-memory BM25 index

This is slower than incremental maintenance but simpler and safer for the first implementation. The chunk corpus is expected to be small enough for this tradeoff.

## Failure Handling

### Query-Time Degradation

- if BM25 is disabled in config, use dense-only retrieval
- if BM25 corpus is empty, use dense-only retrieval
- if BM25 load or search fails, log a warning and use dense-only retrieval
- if reranker fails, keep existing fallback behavior and return original parent order

### Mutation Consistency

Uploads and deletes now affect both Qdrant and BM25 storage. The first version should fail fast when BM25 persistence or rebuild fails instead of silently accepting partial writes.

Operational consequence:

- if Qdrant write succeeds and BM25 update fails, the request should return an error
- recovery can happen through a rebuild path based on persisted files and stored uploads

This is stricter than eventual consistency but easier to reason about and test.

## Configuration Changes

Add the following settings in `app/config.py`:

- `enable_hybrid_retrieval: bool`
- `enable_bm25: bool`
- `dense_child_retrieval_k: int`
- `bm25_child_retrieval_k: int`
- `hybrid_child_fused_k: int`
- `rrf_k: int`
- `bm25_index_path: Path`

Behavioral intent:

- `dense_child_retrieval_k`: dense candidate count before fusion
- `bm25_child_retrieval_k`: BM25 candidate count before fusion
- `hybrid_child_fused_k`: number of fused child chunks kept before parent expansion
- `rrf_k`: damping constant in RRF
- `bm25_index_path`: persisted child-corpus file location

The current `retrieval_top_k` remains the final number of parent chunks returned after reranking.

## Testing Strategy

### Unit Tests

Add focused tests for:

- BM25 corpus load and rebuild from JSON
- tokenization behavior
- add and delete document record flows
- RRF score accumulation and ranking order
- parent aggregation from fused child hits
- dense-only fallback when BM25 is disabled, empty, or failing

### Retrieval Integration Tests

Add tests for the new retriever covering:

- hybrid path with both dense and BM25 results
- duplicate `chunk_id` fusion across both lists
- parent expansion and deduplication
- reranker integration still receiving parent documents
- final `retrieval_top_k` truncation

### API-Level Tests

Add tests around upload and delete behavior:

- upload persists BM25 chunk records
- replacing a file removes old BM25 records before adding new ones
- deleting a file removes both Qdrant data and BM25 records

## Rollout Notes

This design intentionally limits the first version:

- no Qdrant hybrid migration
- no sparse-vector dependency
- no advanced Chinese segmentation pipeline
- no background rebuild worker

That keeps the change narrowly scoped to retrieval quality improvement with clear fallback behavior.

## Open Implementation Decisions

These should be resolved during implementation planning, not by expanding scope:

- exact tokenization rule for Chinese and mixed-language text
- whether BM25 rebuild is triggered synchronously per mutation or through a single shared manager with cache invalidation
- whether mutation rollback helpers are needed for Qdrant/BM25 partial failure recovery

## Accepted Design

The accepted direction is:

- keep Qdrant dense retrieval
- add local BM25 lexical retrieval over child chunks
- fuse dense and BM25 rankings with RRF at child-chunk level
- expand fused results to parents
- preserve current reranker as the final precision stage
