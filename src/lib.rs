#![forbid(unsafe_code)]

//! Bare-metal core primitives for the toy distributed learner.
//!
//! Phase 1 is intentionally small and explicit:
//! - a configurable even-sized, 3-regular graph,
//! - a verified Ramanujan-style topology search,
//! - ECS entities that only see their graph-local neighborhood,
//! - no global tensor passes or backpropagation machinery.

pub mod core_math;
pub mod data;
pub mod ecs_runtime;
pub mod experiment;
pub mod training;

/// Legacy MVP graph size used by the original fixed-topology prototype.
pub const NODE_COUNT: usize = 100;
pub const REGULAR_DEGREE: usize = 3;
pub const MNIST_LABEL_CLASS_COUNT: usize = 10;
pub const CONDITIONED_INPUT_NODE_COUNT: usize =
    data::mnist_loader::MNIST_IMAGE_PIXELS + MNIST_LABEL_CLASS_COUNT;

pub type NodeIndex = usize;
pub type Edge = (NodeIndex, NodeIndex);
