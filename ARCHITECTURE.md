# Architecture & Design Decisions

## 1. Problem shape

Card fraud is a **streaming, imbalanced, latency-sensitive** problem:

* Events arrive in time order and must be scored **before** the transaction is
  authorized — a few milliseconds, not a batch job.
* Fraud is **rare** (~0.5% here) and **bursty** (attacks come in episodes).
* The signal lives almost entirely in **per-entity temporal context** (how this
  user/merchant is behaving *right now* vs. their history), not in the raw event
  fields. So the feature engine is the product; the model is commodity.
* **Leakage is fatal.** A feature that peeks at even one future event inflates
  offline metrics and collapses in production.

## 2. Component map

```
 TransactionStream ──► OnlineFeatureStore ──►  offline table (Parquet)  ──► HistGBDT
 (generator, bounded    (bounded per-entity     (chunked row groups,        (time split,
  per-user state)        ring buffers + LRU)     zstd, float32)              sample weights)
        │                        │                                                │
        │                        └──────────────► online serving ◄───────────────┘
        │                          features_for(event)  +  predict_proba  →  p50/p99 latency
```

Everything is one importable package (`src/fraud/`), so the *same* feature code
builds the offline training table **and** serves online — no train/serve skew.

## 3. Key design decisions

### 3.1 Time-ordered generation, no global sort
Inter-arrival times are drawn from an exponential distribution and the clock only
moves forward, so the stream is **born sorted**. A real system never has the
luxury of sorting an unbounded stream; neither does this one. Fraud is injected as
consecutive **bursts** on a victim user (velocity, geo-impossible, amount anomaly,
account takeover), which is what makes the temporal features light up.

### 3.2 Bounded state is the whole scaling story
The online store keeps, per entity:
* a **ring buffer** (`deque(maxlen=64)`) of recent `(ts, amount, merchant)`, and
* a handful of running scalars (last ts/geo/device, historical max amount).

Per-event work is an O(64) scan of the ring buffer — a small constant — so
throughput is linear in events. The number of *tracked* entities is capped with
**LRU eviction** (`OrderedDict.move_to_end` / `popitem(last=False)`). Result:

> **RSS is O(entities), not O(events).** Streaming 1B events costs the same memory
> as streaming 1M. This is the property that makes 1B a throughput/sharding
> problem, not a memory problem.

### 3.3 Leakage safety by construction
`OnlineFeatureStore.process()` computes the feature vector from state reflecting
**only past events**, and *then* folds the current event into state. The event can
never appear in its own aggregates. Offline, training uses a **chronological
split** (train on the earlier stream, test on the later) rather than a random
split. Two tests lock this down: an ordering test and a whole-pipeline brute-force
recompute that only ever looks at `ts < ts_i`.

### 3.4 Model: HistGradientBoostingClassifier
Histogram GBDT is the right default for tabular fraud: native handling of skewed
numeric features, monotone-ish decision boundaries, fast CPU inference (sub-ms per
row), and no scaling/imputation pipeline to drift. Imbalance is handled with
**balanced sample weights** (each class contributes equal total weight) so the
model optimizes the minority class without resampling. Early stopping on an
internal validation slice prevents overfitting the tiny positive set.

### 3.5 Metrics that match the job
Under 0.5% prevalence, accuracy and ROC-AUC are misleading. The headline is
**PR-AUC (average precision)** with the prevalence rate as the honest baseline
(a random classifier's PR-AUC *equals* the base rate). Operationally we report
**precision@k / alert precision** at a fixed alert budget (the fraction of the
stream an ops team can review), **fraud-caught %** at that budget, and a
**calibration** curve.

## 4. Scaling to 1B — the extrapolation

Measured here: **5M events streamed single-core** with flat RSS (see
`benchmarks/scaling_results.csv`, `make bench`). The path to 1B is horizontal,
not vertical:

| Concern | This repo (single process) | Production analog @ 1B/day |
|---|---|---|
| Transport | in-process generator | **Kafka** topic, partitioned by `user_id` |
| Stateful compute | `OnlineFeatureStore` (LRU dicts) | **Flink** / Spark Structured Streaming keyed state |
| State store | Python dict ring buffers | **RocksDB**-backed keyed state / Redis feature cache |
| Sharding | one core | hash-partition by `user_id` → N parallel operators, each owning a disjoint key range (state stays bounded per shard) |
| Offline table | chunked Parquet (zstd) | partitioned Parquet on object storage; DuckDB/Spark for training reads |
| Serving | `predict_proba` on 1 row | same model behind a gRPC scorer; feature read from the keyed state store |

Because state is **partitioned by entity**, every shard holds a bounded slice of
users and the per-event cost is unchanged. Throughput scales ~linearly with shard
count: at the measured single-core rate, 1B events is hours single-core and
minutes across a modest cluster. No design element grows with the total event
count — only with the number of *active entities*, which is itself bounded.

### 4.1 Watermarks & late data
The generator emits in-order, so this repo doesn't need watermarking. In
production, Kafka partitions can interleave; the Flink analog would use
**event-time windows with a bounded-lateness watermark**, and the ring-buffer
window scan already tolerates small out-of-order deltas (it filters on `dt >= 0`).

## 5. Determinism
Every stochastic component is seeded (`SEED = 42`): generator RNG, model
`random_state`, permutation-importance RNG. Re-running `make run` reproduces the
same table, model, metrics, and screenshots.
