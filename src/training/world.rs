use hecs::{Entity, NoSuchEntity, World};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use thiserror::Error;

use crate::core_math::ramanujan_gen::VerifiedRamanujanGraph;
use crate::data::mnist_loader::MNIST_IMAGE_PIXELS;
use crate::ecs_runtime::components::{
    InputNode, LocalWeights, NodeState, StableNodeIndex, TopologyPointers,
};
use crate::REGULAR_DEGREE;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldSummary {
    pub mean_abs_activation: f32,
    pub mean_squared_activation: f32,
    pub mean_abs_weight: f32,
}

#[derive(Debug, Error)]
pub enum WorldBuildError {
    #[error("graph edge ({u}, {v}) references an out-of-bounds node for node_count {node_count}")]
    InvalidGraphEdge {
        u: usize,
        v: usize,
        node_count: usize,
    },
    #[error("graph node {node} has {actual} neighbors, expected {expected}")]
    InvalidNeighborCount {
        node: usize,
        actual: usize,
        expected: usize,
    },
    #[error("failed to attach components to entity for graph node {node}: {source}")]
    EntityInsertion {
        node: usize,
        #[source]
        source: NoSuchEntity,
    },
}

pub fn build_training_world(
    graph: &VerifiedRamanujanGraph,
    weight_seed: u64,
    weight_init_scale: f32,
    input_node_count: usize,
) -> Result<World, WorldBuildError> {
    let node_count = graph.certificate.node_count;
    let adjacency = build_adjacency_lists(node_count, &graph.edges)?;
    let mut world = World::new();
    let entities: Vec<Entity> = (0..node_count)
        .map(|node_index| {
            if node_index < input_node_count {
                world.spawn((
                    InputNode,
                    StableNodeIndex::new(node_index),
                    NodeState::new(0.0),
                ))
            } else {
                world.spawn((StableNodeIndex::new(node_index), NodeState::new(0.0)))
            }
        })
        .collect();
    let mut rng = ChaCha8Rng::seed_from_u64(weight_seed);

    for (node_index, entity) in entities.iter().copied().enumerate() {
        let neighbors: [usize; REGULAR_DEGREE] = adjacency[node_index]
            .as_slice()
            .try_into()
            .map_err(|_| WorldBuildError::InvalidNeighborCount {
                node: node_index,
                actual: adjacency[node_index].len(),
                expected: REGULAR_DEGREE,
            })?;
        let topology =
            TopologyPointers::new(neighbors.map(|neighbor_index| entities[neighbor_index]));
        let weights = LocalWeights::new(std::array::from_fn(|_| {
            rng.gen_range(-weight_init_scale..=weight_init_scale)
        }));

        world
            .insert(entity, (weights, topology))
            .map_err(|source| WorldBuildError::EntityInsertion {
                node: node_index,
                source,
            })?;
    }

    Ok(world)
}

pub fn training_input_node_count(graph_node_count: usize) -> usize {
    graph_node_count.min(MNIST_IMAGE_PIXELS)
}

pub fn count_nodes(world: &World) -> usize {
    world.query::<&NodeState>().iter().count()
}

pub fn count_input_nodes(world: &World) -> usize {
    world.query::<&InputNode>().iter().count()
}

pub fn summarize_world(world: &World) -> WorldSummary {
    let mut activation_count = 0usize;
    let mut abs_activation_sum = 0.0f32;
    let mut squared_activation_sum = 0.0f32;

    for (_entity, state) in world.query::<&NodeState>().iter() {
        activation_count += 1;
        abs_activation_sum += state.activation.abs();
        squared_activation_sum += state.activation * state.activation;
    }

    let mut weight_count = 0usize;
    let mut abs_weight_sum = 0.0f32;
    for (_entity, weights) in world.query::<&LocalWeights>().iter() {
        for weight in weights.neighbor_weights {
            weight_count += 1;
            abs_weight_sum += weight.abs();
        }
    }

    WorldSummary {
        mean_abs_activation: if activation_count == 0 {
            0.0
        } else {
            abs_activation_sum / activation_count as f32
        },
        mean_squared_activation: if activation_count == 0 {
            0.0
        } else {
            squared_activation_sum / activation_count as f32
        },
        mean_abs_weight: if weight_count == 0 {
            0.0
        } else {
            abs_weight_sum / weight_count as f32
        },
    }
}

fn build_adjacency_lists(
    node_count: usize,
    edges: &[(usize, usize)],
) -> Result<Vec<Vec<usize>>, WorldBuildError> {
    let mut adjacency = vec![Vec::with_capacity(REGULAR_DEGREE); node_count];

    for &(u, v) in edges {
        if u >= node_count || v >= node_count {
            return Err(WorldBuildError::InvalidGraphEdge { u, v, node_count });
        }

        adjacency[u].push(v);
        adjacency[v].push(u);
    }

    for (node, neighbors) in adjacency.iter_mut().enumerate() {
        neighbors.sort_unstable();
        if neighbors.len() != REGULAR_DEGREE {
            return Err(WorldBuildError::InvalidNeighborCount {
                node,
                actual: neighbors.len(),
                expected: REGULAR_DEGREE,
            });
        }
    }

    Ok(adjacency)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_input_node_count_caps_at_mnist_pixels() {
        assert_eq!(training_input_node_count(100), 100);
        assert_eq!(
            training_input_node_count(MNIST_IMAGE_PIXELS),
            MNIST_IMAGE_PIXELS
        );
        assert_eq!(
            training_input_node_count(MNIST_IMAGE_PIXELS + 64),
            MNIST_IMAGE_PIXELS
        );
    }
}
