//! ECS component definitions for the local-only learner.
//!
//! Each graph node is its own ECS entity. The topology is intentionally encoded as
//! fixed-size arrays because the phase-1 graph degree is constant (`d = 3`), which
//! keeps memory layout compact and makes the correspondence between topology slots and
//! local weights explicit.

use hecs::Entity;

use crate::REGULAR_DEGREE;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeState {
    /// Current scalar activation visible to graph-local neighbors.
    pub activation: f32,
}

impl NodeState {
    pub fn new(activation: f32) -> Self {
        Self { activation }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InputNode;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalWeights {
    /// Weight slot `i` multiplies the activation coming from `TopologyPointers::neighbors[i]`.
    pub neighbor_weights: [f32; REGULAR_DEGREE],
}

impl LocalWeights {
    pub fn new(neighbor_weights: [f32; REGULAR_DEGREE]) -> Self {
        Self { neighbor_weights }
    }

    pub fn filled(value: f32) -> Self {
        Self {
            neighbor_weights: [value; REGULAR_DEGREE],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopologyPointers {
    /// ECS entity IDs for the node's exactly-three graph neighbors.
    ///
    /// The ordering matters: `neighbors[i]` is wired to `LocalWeights::neighbor_weights[i]`.
    pub neighbors: [Entity; REGULAR_DEGREE],
}

impl TopologyPointers {
    pub fn new(neighbors: [Entity; REGULAR_DEGREE]) -> Self {
        Self { neighbors }
    }
}
