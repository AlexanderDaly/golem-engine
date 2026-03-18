#![forbid(unsafe_code)]

//! Bare-metal core primitives for the toy distributed learner.
//!
//! Phase 1 is intentionally small and explicit:
//! - a fixed 100-node, 3-regular graph,
//! - a verified Ramanujan-style topology search,
//! - ECS entities that only see their graph-local neighborhood,
//! - no global tensor passes or backpropagation machinery.

pub mod core_math;
pub mod ecs_runtime;

pub const NODE_COUNT: usize = 100;
pub const REGULAR_DEGREE: usize = 3;

pub type NodeIndex = usize;
pub type Edge = (NodeIndex, NodeIndex);
