//! Checkpoint serialization for ECS graph state.
//!
//! The runtime uses `hecs::Entity` values for in-memory topology pointers, but those IDs are
//! not stable across process boundaries. This module snapshots the graph through explicit
//! stable node indices so a saved world can be reconstructed exactly on load.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use hecs::{Entity, NoSuchEntity, World};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ecs_runtime::components::{
    InputNode, LocalWeights, NodeState, StableNodeIndex, TopologyPointers,
};
use crate::ecs_runtime::ingestion_system::SimulationPhase;
use crate::{NodeIndex, REGULAR_DEGREE};

pub const CHECKPOINT_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphCheckpoint {
    pub format_version: u32,
    pub phase: SimulationPhase,
    pub nodes: Vec<GraphNodeCheckpoint>,
}

impl GraphCheckpoint {
    pub fn from_world(world: &World, phase: SimulationPhase) -> Result<Self, CheckpointError> {
        let graph_entities: Vec<Entity> = world
            .query::<&TopologyPointers>()
            .iter()
            .map(|(entity, _)| entity)
            .collect();

        let mut nodes = Vec::with_capacity(graph_entities.len());
        for entity in graph_entities {
            let stable_index = world
                .get::<&StableNodeIndex>(entity)
                .map_err(|_| CheckpointError::MissingStableNodeIndex { node: entity })?
                .index;
            let activation = world
                .get::<&NodeState>(entity)
                .map_err(|_| CheckpointError::MissingState { node: entity })?
                .activation;
            let local_weights = *world
                .get::<&LocalWeights>(entity)
                .map_err(|_| CheckpointError::MissingWeights { node: entity })?;
            let topology = world
                .get::<&TopologyPointers>(entity)
                .map_err(|_| CheckpointError::MissingTopology { node: entity })?;

            let mut neighbor_indices = [0usize; REGULAR_DEGREE];
            for (slot, neighbor) in topology.neighbors.iter().copied().enumerate() {
                neighbor_indices[slot] = world
                    .get::<&StableNodeIndex>(neighbor)
                    .map_err(|_| CheckpointError::MissingStableNodeIndex { node: neighbor })?
                    .index;
            }

            nodes.push(GraphNodeCheckpoint {
                index: stable_index,
                activation,
                is_input: world.get::<&InputNode>(entity).is_ok(),
                local_weights,
                topology: StableTopologyPointers::new(neighbor_indices),
            });
        }

        validate_serialized_nodes(&mut nodes)?;

        Ok(Self {
            format_version: CHECKPOINT_FORMAT_VERSION,
            phase,
            nodes,
        })
    }

    pub fn into_world(self) -> Result<(World, SimulationPhase), CheckpointError> {
        self.validate_version()?;

        let mut nodes = self.nodes;
        validate_serialized_nodes(&mut nodes)?;

        let mut world = World::new();
        let entities: Vec<Entity> = nodes
            .iter()
            .map(|node| {
                if node.is_input {
                    world.spawn((
                        InputNode,
                        StableNodeIndex::new(node.index),
                        NodeState::new(node.activation),
                    ))
                } else {
                    world.spawn((
                        StableNodeIndex::new(node.index),
                        NodeState::new(node.activation),
                    ))
                }
            })
            .collect();

        for node in &nodes {
            let neighbors = node.topology.resolve_entities(&entities)?;
            let entity = entities[node.index];
            let topology = TopologyPointers::new(neighbors);

            world
                .insert(entity, (node.local_weights, topology))
                .map_err(|source| CheckpointError::EntityInsertion {
                    node: node.index,
                    source,
                })?;
        }

        Ok((world, self.phase))
    }

    fn validate_version(&self) -> Result<(), CheckpointError> {
        if self.format_version == CHECKPOINT_FORMAT_VERSION {
            Ok(())
        } else {
            Err(CheckpointError::UnsupportedFormatVersion {
                found: self.format_version,
                expected: CHECKPOINT_FORMAT_VERSION,
            })
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphNodeCheckpoint {
    pub index: NodeIndex,
    pub activation: f32,
    pub is_input: bool,
    pub local_weights: LocalWeights,
    pub topology: StableTopologyPointers,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StableTopologyPointers {
    pub neighbors: [NodeIndex; REGULAR_DEGREE],
}

impl StableTopologyPointers {
    pub fn new(neighbors: [NodeIndex; REGULAR_DEGREE]) -> Self {
        Self { neighbors }
    }

    fn resolve_entities(
        self,
        entities: &[Entity],
    ) -> Result<[Entity; REGULAR_DEGREE], CheckpointError> {
        let mut resolved = [Entity::DANGLING; REGULAR_DEGREE];
        for (slot, neighbor_index) in self.neighbors.iter().copied().enumerate() {
            resolved[slot] =
                *entities
                    .get(neighbor_index)
                    .ok_or(CheckpointError::UnknownNeighborIndex {
                        node: None,
                        neighbor_index,
                        node_count: entities.len(),
                    })?;
        }

        Ok(resolved)
    }
}

pub fn save_checkpoint_json<P: AsRef<Path>>(
    path: P,
    world: &World,
    phase: SimulationPhase,
) -> Result<(), CheckpointError> {
    let checkpoint = GraphCheckpoint::from_world(world, phase)?;
    let path = path.as_ref().to_path_buf();
    let bytes =
        serde_json::to_vec_pretty(&checkpoint).map_err(|source| CheckpointError::Serialize {
            path: path.clone(),
            source,
        })?;

    fs::write(&path, bytes).map_err(|source| CheckpointError::Io {
        path: path.clone(),
        source,
    })
}

pub fn load_checkpoint_json<P: AsRef<Path>>(
    path: P,
) -> Result<(World, SimulationPhase), CheckpointError> {
    let checkpoint = read_checkpoint_json(path)?;
    checkpoint.into_world()
}

pub fn read_checkpoint_json<P: AsRef<Path>>(path: P) -> Result<GraphCheckpoint, CheckpointError> {
    let path = path.as_ref().to_path_buf();
    let bytes = fs::read(&path).map_err(|source| CheckpointError::Io {
        path: path.clone(),
        source,
    })?;

    serde_json::from_slice(&bytes).map_err(|source| CheckpointError::Deserialize { path, source })
}

fn validate_serialized_nodes(nodes: &mut Vec<GraphNodeCheckpoint>) -> Result<(), CheckpointError> {
    nodes.sort_unstable_by_key(|node| node.index);

    for (expected_index, node) in nodes.iter().enumerate() {
        if node.index != expected_index {
            return Err(CheckpointError::SparseOrOutOfOrderNodeIndices {
                expected: expected_index,
                found: node.index,
            });
        }

        for neighbor_index in node.topology.neighbors {
            if neighbor_index >= nodes.len() {
                return Err(CheckpointError::UnknownNeighborIndex {
                    node: Some(node.index),
                    neighbor_index,
                    node_count: nodes.len(),
                });
            }
        }
    }

    Ok(())
}

#[derive(Debug, Error)]
pub enum CheckpointError {
    #[error("failed to read or write checkpoint {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to serialize checkpoint {path}: {source}")]
    Serialize {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to deserialize checkpoint {path}: {source}")]
    Deserialize {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("unsupported checkpoint format version {found}; expected {expected}")]
    UnsupportedFormatVersion { found: u32, expected: u32 },
    #[error("graph node {node:?} is missing a StableNodeIndex component")]
    MissingStableNodeIndex { node: Entity },
    #[error("graph node {node:?} is missing a NodeState component")]
    MissingState { node: Entity },
    #[error("graph node {node:?} is missing a LocalWeights component")]
    MissingWeights { node: Entity },
    #[error("graph node {node:?} is missing a TopologyPointers component")]
    MissingTopology { node: Entity },
    #[error(
        "checkpoint node indices must be dense and zero-based; expected {expected}, found {found}"
    )]
    SparseOrOutOfOrderNodeIndices { expected: usize, found: usize },
    #[error("checkpoint node {node:?} references neighbor index {neighbor_index}, but node_count is {node_count}")]
    UnknownNeighborIndex {
        node: Option<NodeIndex>,
        neighbor_index: NodeIndex,
        node_count: usize,
    },
    #[error("failed to attach components to reconstructed node {node}: {source}")]
    EntityInsertion {
        node: NodeIndex,
        #[source]
        source: NoSuchEntity,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs_runtime::components::{
        LocalWeights, NodeState, StableNodeIndex, TopologyPointers,
    };

    #[test]
    fn checkpoint_round_trip_preserves_topology_weights_and_phase() {
        let (world, phase) = sample_world();

        let checkpoint = GraphCheckpoint::from_world(&world, phase).expect("serialize world");
        let json = serde_json::to_string(&checkpoint).expect("serialize checkpoint json");
        let checkpoint: GraphCheckpoint =
            serde_json::from_str(&json).expect("deserialize checkpoint json");
        let expected_checkpoint = checkpoint.clone();
        let (restored_world, restored_phase) = checkpoint.into_world().expect("restore world");

        assert_eq!(restored_phase, phase);

        let restored = GraphCheckpoint::from_world(&restored_world, restored_phase)
            .expect("re-serialize restored world");
        assert_eq!(restored, expected_checkpoint);
    }

    #[test]
    fn checkpoint_fails_when_stable_indices_are_sparse() {
        let mut world = World::new();
        let first = world.spawn((
            InputNode,
            StableNodeIndex::new(0),
            NodeState::new(0.0),
            LocalWeights::filled(0.0),
        ));
        let second = world.spawn((
            InputNode,
            StableNodeIndex::new(2),
            NodeState::new(0.0),
            LocalWeights::filled(0.0),
        ));

        world
            .insert(first, (TopologyPointers::new([first, first, first]),))
            .expect("attach first topology");
        world
            .insert(second, (TopologyPointers::new([first, first, first]),))
            .expect("attach second topology");

        let error = GraphCheckpoint::from_world(&world, SimulationPhase::Positive)
            .expect_err("sparse indices should fail");

        assert!(matches!(
            error,
            CheckpointError::SparseOrOutOfOrderNodeIndices {
                expected: 1,
                found: 2
            }
        ));
    }

    fn sample_world() -> (World, SimulationPhase) {
        let mut world = World::new();
        let first = world.spawn((
            InputNode,
            StableNodeIndex::new(0),
            NodeState::new(0.25),
            LocalWeights::new([0.1, 0.2, 0.3]),
        ));
        let second = world.spawn((
            StableNodeIndex::new(1),
            NodeState::new(-0.75),
            LocalWeights::new([-0.4, 0.5, -0.6]),
        ));
        let third = world.spawn((
            InputNode,
            StableNodeIndex::new(2),
            NodeState::new(1.5),
            LocalWeights::new([0.7, -0.8, 0.9]),
        ));

        world
            .insert(first, (TopologyPointers::new([second, third, second]),))
            .expect("attach first topology");
        world
            .insert(second, (TopologyPointers::new([first, third, first]),))
            .expect("attach second topology");
        world
            .insert(third, (TopologyPointers::new([first, second, first]),))
            .expect("attach third topology");

        (world, SimulationPhase::Negative)
    }
}
