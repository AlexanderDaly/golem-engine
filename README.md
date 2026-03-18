# Golem Engine

Golem Engine is the Rust core for Project Sephirot, a toy decentralized learning system built around a sparse asynchronous graph instead of dense backpropagation. The current implementation targets a fixed 100-node, 3-regular Ramanujan-style topology and an ECS runtime where each node updates from only its local neighborhood.

This repository is intentionally narrow in scope. It focuses on the inspectable systems needed to bootstrap the architecture:

- deterministic generation and validation of a `d = 3`, `n = 100` graph,
- ECS-native node state and local update dynamics,
- MNIST ingestion for Forward-Forward style contrastive training,
- procedurally generated negative samples formed by hybridizing real digits.

## Design Principles

- No backpropagation or dense tensor passes.
- Locality first: each entity reads only graph-local state.
- Explicit data flow over framework magic.
- Dependency-light file ingestion and memory-safe Rust throughout.

## Current Architecture

### 1. Ramanujan Graph Generation

The graph generator in [`src/core_math/ramanujan_gen.rs`](/Users/alexanderdaly/Projects/golem-engine/src/core_math/ramanujan_gen.rs) builds candidate 3-regular graphs from edge-disjoint perfect matchings, validates simplicity and connectivity, and computes the adjacency spectrum directly with `nalgebra`. A graph is accepted only if it satisfies the current Ramanujan bound.

### 2. ECS Runtime

The ECS layer uses `hecs` and models each graph node as its own entity.

- [`NodeState`](/Users/alexanderdaly/Projects/golem-engine/src/ecs_runtime/components.rs#L13) stores the node’s current scalar activation.
- [`LocalWeights`](/Users/alexanderdaly/Projects/golem-engine/src/ecs_runtime/components.rs#L27) stores the three edge-local weights.
- [`TopologyPointers`](/Users/alexanderdaly/Projects/golem-engine/src/ecs_runtime/components.rs#L45) stores exactly three neighbor entity IDs.
- [`update_nodes_forward_forward`](/Users/alexanderdaly/Projects/golem-engine/src/ecs_runtime/systems.rs#L47) performs an in-place asynchronous sweep, so later entities in the tick may observe earlier updates.
- [`update_local_weights_forward_forward`](/Users/alexanderdaly/Projects/golem-engine/src/ecs_runtime/systems.rs) applies a local Forward-Forward learning step by scoring neighborhood goodness and nudging only the node’s own incoming weights.

### 3. Contrastive MNIST Pipeline

The data path is split into three pieces:

- [`MnistDataset`](/Users/alexanderdaly/Projects/golem-engine/src/data/mnist_loader.rs#L27) loads raw IDX files into contiguous byte buffers and exposes normalized 784-float views of images.
- [`generate_hybrid_negative`](/Users/alexanderdaly/Projects/golem-engine/src/data/procedural_negatives.rs#L29) creates plausible negatives by stitching two distinct digits with a softened seam across the midpoint.
- [`inject_data_system`](/Users/alexanderdaly/Projects/golem-engine/src/ecs_runtime/ingestion_system.rs#L143) runs as an ECS ingestion step, alternating between positive and negative samples by tracking [`SimulationPhase`](/Users/alexanderdaly/Projects/golem-engine/src/ecs_runtime/ingestion_system.rs#L24).

## Important Constraint

MNIST images have 784 pixels, but the current graph has 100 scalar nodes. The ingestion system therefore projects contiguous spans of the flattened image onto the available input entities when fewer than 784 `InputNode`s are present. If the graph later exposes 784 input entities, the same system already supports direct one-pixel-per-node injection.

## Repository Layout

```text
src/
  core_math/
    ramanujan_gen.rs        Graph generation, validation, spectral certificate
  data/
    mnist_loader.rs         Native IDX parsing and normalization
    procedural_negatives.rs Hybrid negative generation
  ecs_runtime/
    components.rs           Node ECS components and input marker
    ingestion_system.rs     Contrastive data injection and phase toggling
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

The crate is currently a library, not a CLI application, so `cargo test` is the main validation path.

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
- ECS node model for graph-local asynchronous updates,
- MNIST positive stream with native IDX parsing,
- plausible negative generation via digit hybridization,
- phase-driven ingestion into input entities,
- node-local Forward-Forward goodness evaluation and weight updates.

Not yet implemented:

- a training loop binary,
- checkpointing or experiment management,
- distributed runtime across multiple physical workers.

## License

No license file is present yet. Until one is added, treat the repository as all rights reserved.
