# Golem Engine

Golem Engine is the Rust core for Project Sephirot, a toy decentralized learning system built around a sparse asynchronous graph instead of dense backpropagation. The current implementation targets configurable 3-regular Ramanujan-style topologies and an ECS runtime where each node updates from only its local neighborhood.

This repository is intentionally narrow in scope. It focuses on the inspectable systems needed to bootstrap the architecture:

- deterministic generation and validation of configurable `d = 3` regular graphs,
- ECS-native node state and local update dynamics,
- label-conditioned MNIST ingestion for Forward-Forward style contrastive training,
- run-directory experiment management with manifest, metrics, and checkpoints.

## Design Principles

- No backpropagation or dense tensor passes.
- Locality first: each entity reads only graph-local state.
- Explicit data flow over framework magic.
- Dependency-light file ingestion and memory-safe Rust throughout.

## Current Architecture

### 1. Ramanujan Graph Generation

The graph generator in [`src/core_math/ramanujan_gen.rs`](/Users/alexanderdaly/Projects/golem-engine/src/core_math/ramanujan_gen.rs) builds candidate 3-regular graphs from edge-disjoint perfect matchings, validates simplicity and connectivity, uses a matrix-free proxy filter for larger searches, and computes the exact adjacency spectrum with `nalgebra` for the candidates that survive. A graph is accepted only if it satisfies the current Ramanujan bound.

### 2. ECS Runtime

The ECS layer uses `hecs` and models each graph node as its own entity.

- [`NodeState`](/Users/alexanderdaly/Projects/golem-engine/src/ecs_runtime/components.rs#L13) stores the node’s current scalar activation.
- [`LocalWeights`](/Users/alexanderdaly/Projects/golem-engine/src/ecs_runtime/components.rs#L27) stores the three edge-local weights.
- [`TopologyPointers`](/Users/alexanderdaly/Projects/golem-engine/src/ecs_runtime/components.rs#L45) stores exactly three neighbor entity IDs.
- [`update_nodes_forward_forward`](/Users/alexanderdaly/Projects/golem-engine/src/ecs_runtime/systems.rs#L47) performs an in-place asynchronous sweep, so later entities in the tick may observe earlier updates.
- [`update_local_weights_forward_forward`](/Users/alexanderdaly/Projects/golem-engine/src/ecs_runtime/systems.rs) applies a local Forward-Forward learning step by scoring neighborhood goodness and nudging only the node’s own incoming weights.

### 3. Label-Conditioned MNIST Pipeline

The data path is split into three pieces:

- [`MnistDataset`](/Users/alexanderdaly/Projects/golem-engine/src/data/mnist_loader.rs#L27) loads raw IDX files into contiguous byte buffers and exposes normalized 784-float views of images.
- [`fill_positive_sample`](/Users/alexanderdaly/Projects/golem-engine/src/ecs_runtime/ingestion_system.rs) prepares real digits with the true candidate label.
- [`fill_negative_sample`](/Users/alexanderdaly/Projects/golem-engine/src/ecs_runtime/ingestion_system.rs) reuses the same real image but swaps in a deterministic wrong label.
- [`inject_data_system`](/Users/alexanderdaly/Projects/golem-engine/src/ecs_runtime/ingestion_system.rs) writes pixels into input slots `0..783`, writes a one-hot candidate label into slots `784..793`, and toggles [`SimulationPhase`](/Users/alexanderdaly/Projects/golem-engine/src/ecs_runtime/ingestion_system.rs#L24).

## Important Constraint

MNIST images have 784 pixels, and label-conditioned Forward-Forward reserves 10 additional input slots for one-hot digit labels. Fresh runs therefore require at least 794 graph nodes, and the training binary now defaults to a 794-node graph so MNIST pixels and label slots are both available out of the box. Resuming old 784-input checkpoints is rejected with a compatibility error.

## Repository Layout

```text
src/
  core_math/
    ramanujan_gen.rs        Graph generation, validation, spectral certificate
  data/
    mnist_loader.rs         Native IDX parsing and normalization
    procedural_negatives.rs Legacy hybrid-negative utilities
  ecs_runtime/
    components.rs           Node ECS components and input marker
    ingestion_system.rs     Label-conditioned data injection and phase toggling
    systems.rs              Local asynchronous node updates
  lib.rs                    Crate exports and global constants
```

## Getting Started

### Prerequisites

- Rust stable toolchain
- Cargo

### Build and Test

```bash
cargo test
```

The crate now ships with a training binary as well as the library API. `cargo test` remains the main correctness check for the reusable components.

### Run The Training Loop

```bash
cargo run -- \
  --train-images data/train-images-idx3-ubyte \
  --train-labels data/train-labels-idx1-ubyte \
  --test-images data/t10k-images-idx3-ubyte \
  --test-labels data/t10k-labels-idx1-ubyte \
  --epochs 5 \
  --learning-rate 0.001 \
  --activation tanh \
  --graph-node-count 794 \
  --eval-every 1 \
  --checkpoint-every 1
```

Useful options:

- `--graph-node-count` controls the graph size. Fresh label-conditioned runs require at least `794`, and the binary defaults to `794` so the input surface includes `784` pixel slots plus `10` label slots.
- `--graph-search-limit` controls how many deterministic seeds are searched when generating the verified Ramanujan graph. If omitted, the default search budget scales with the requested node count.
- `--weight-seed` and `--weight-init-scale` control the initial local weights attached to each node.
- `--run-dir` overrides the default `runs/<unix-timestamp>-<short-id>/` experiment directory.
- `--eval-every` runs a no-learning Forward-Forward evaluation pass every `N` epochs and writes the results into `metrics.jsonl`.
- `--checkpoint-every` saves `checkpoints/epoch-<N>.json` inside the run directory at the requested interval and on the final epoch.
- `--save-checkpoint` writes an additional final checkpoint to a user-specified path.
- `--load-checkpoint` resumes from a previously saved checkpoint instead of generating a fresh graph and weight initialization.
- `--distributed-worker <host:port>` enables distributed training and evaluation by sending per-epoch work shards to one or more remote worker processes.

The binary constructs the verified sparse graph, spawns one ECS entity per graph node, tags the first `794` nodes as `InputNode`, and runs the per-tick schedule:

1. `inject_data_system`
2. `update_nodes_forward_forward`
3. `update_local_weights_forward_forward`

One epoch is defined as `2 * dataset_len` ticks so that the positive and negative cursors each traverse the dataset once.

Forward-Forward evaluation uses the same sparse forward sweep without any weight updates. For each sample, the evaluator resets node activations, sweeps all candidate labels `0..9`, injects the conditioned sample, runs one forward sweep, and predicts the digit from the highest world goodness. `metrics.jsonl` records accuracy together with correct-label goodness, best-wrong-label goodness, and the resulting margin.

Checkpoint files store stable node indices rather than raw `hecs::Entity` IDs, so the graph can be reconstructed exactly across process restarts.

### Run Distributed Workers

Start one worker process on each machine that should execute shards:

```bash
cargo run -- --worker-listen 0.0.0.0:7000
```

Then point the coordinator at those workers:

```bash
cargo run -- \
  --train-images data/train-images-idx3-ubyte \
  --train-labels data/train-labels-idx1-ubyte \
  --test-images data/t10k-images-idx3-ubyte \
  --test-labels data/t10k-labels-idx1-ubyte \
  --epochs 5 \
  --distributed-worker worker-a:7000 \
  --distributed-worker worker-b:7000
```

Distributed execution currently uses synchronous data-parallel federated averaging at the epoch boundary:

1. the coordinator snapshots the current world into the existing JSON checkpoint format,
2. each worker receives the same world snapshot plus a disjoint dataset shard,
3. every worker runs the normal local Forward-Forward tick schedule on its shard,
4. the coordinator averages the returned activations and local weights elementwise,
5. distributed evaluation reuses the same worker pool by partitioning dataset indices and aggregating the returned accuracy and goodness sums.

Important constraints:

- every worker must be able to read the same dataset paths passed to the coordinator, or equivalent paths mounted at the same location,
- the current implementation is an epoch-level approximation of the single-process asynchronous schedule, not a fully partitioned cross-machine ECS world,
- checkpoints, manifests, and metrics are still written only by the coordinator.

### Run Directory Layout

By default, each invocation creates a run directory like `runs/1710000000-abc123/`.

```text
runs/
  1710000000-abc123/
    manifest.json
    metrics.jsonl
    checkpoints/
      epoch-000001.json
      epoch-000005.json
```

- `manifest.json` records the effective training configuration, dataset paths, checkpoint settings, conditioning mode, distributed-worker settings, and any resume source.
- `metrics.jsonl` appends one JSON record per epoch with activation, weight, training-phase goodness-separation metrics, and label-conditioned evaluation metrics. If an epoch skips evaluation because of `--eval-every`, the evaluation fields are `null`.
- `checkpoints/epoch-<N>.json` stores periodic or final in-run checkpoints using the existing ECS checkpoint format.

Resume behavior:

- If `--run-dir` is omitted and `--load-checkpoint` points to `runs/.../checkpoints/epoch-<N>.json`, the binary reuses that run directory when doing so would not overwrite newer metrics.
- If the checkpoint is older than the latest metrics already recorded in that run, the binary creates a new run directory and records the checkpoint path in `manifest.json`.
- If you want to keep appending to an existing run while resuming from a checkpoint saved outside the run directory, pass `--run-dir` explicitly.

## Using the MNIST Loader

Download the raw MNIST IDX files and point the loader at them directly.

Typical filenames:

- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

Example:

```rust
use golem_engine::data::mnist_loader::MnistDataset;

let dataset = MnistDataset::load(
    "data/train-images-idx3-ubyte",
    "data/train-labels-idx1-ubyte",
)?;

let first = dataset.normalized_image(0)?;
assert_eq!(first.pixels.len(), 784);
```

## Expected ECS Scheduling

The intended tick order is:

1. Run [`inject_data_system`](/Users/alexanderdaly/Projects/golem-engine/src/ecs_runtime/ingestion_system.rs#L143) at the start of the tick.
2. Run [`update_nodes_forward_forward`](/Users/alexanderdaly/Projects/golem-engine/src/ecs_runtime/systems.rs#L47) to propagate activations through the graph.
3. Run [`update_local_weights_forward_forward`](/Users/alexanderdaly/Projects/golem-engine/src/ecs_runtime/systems.rs) with the phase corresponding to the sample that just propagated. Because [`inject_data_system`](/Users/alexanderdaly/Projects/golem-engine/src/ecs_runtime/ingestion_system.rs#L143) advances the phase immediately, that is the pre-toggle phase, or `phase.toggled()` if you are reusing the post-ingestion phase variable.

The repo now includes the graph primitive, the data path, the asynchronous ECS propagation layer, and a local Forward-Forward weight update step built directly on node-local goodness.

## Status

Implemented:

- verified cubic graph search and spectral certification,
- configurable graph sizes up through MNIST-scale conditioned input mapping,
- ECS node model for graph-local asynchronous updates,
- MNIST positive stream with native IDX parsing,
- deterministic wrong-label negatives on real digits,
- phase-driven label-conditioned ingestion into input entities,
- node-local Forward-Forward goodness evaluation and weight updates,
- label-sweep evaluation with accuracy and goodness-margin reporting,
- run-directory manifests and per-epoch metrics JSONL output,
- periodic checkpointing plus checkpoint resume metadata,
- JSON checkpoint save/load for graph state persistence,
- distributed worker-server mode with per-epoch federated averaging across physical workers.

Not yet implemented:

- true cross-machine partitioning of a single asynchronous ECS world without epoch-boundary averaging.

## License

No license file is present yet. Until one is added, treat the repository as all rights reserved.
