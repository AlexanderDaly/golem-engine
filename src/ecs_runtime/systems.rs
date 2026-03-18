//! ECS systems for the forward-forward style local dynamics.
//!
//! The central architectural rule for this file is that updates remain node-local:
//! - each entity reads only the activations stored on its own topological neighbors,
//! - each entity applies only its own local edge weights,
//! - each entity writes back only its own activation,
//! - writes happen immediately, so later entities in the same tick can observe earlier
//!   updates and there is no hidden global synchronization barrier.

use hecs::{Entity, World};
use thiserror::Error;

use crate::ecs_runtime::components::{LocalWeights, NodeState, TopologyPointers};

#[derive(Debug, Clone, Copy)]
pub enum ActivationKind {
    Tanh,
    Relu,
    SoftSign,
}

impl ActivationKind {
    pub fn apply(self, input: f32) -> f32 {
        match self {
            Self::Tanh => input.tanh(),
            Self::Relu => input.max(0.0),
            Self::SoftSign => input / (1.0 + input.abs()),
        }
    }
}

#[derive(Debug, Error)]
pub enum NodeUpdateError {
    #[error("node {node:?} is missing a NodeState component")]
    MissingState { node: Entity },
    #[error("node {node:?} is missing a LocalWeights component")]
    MissingWeights { node: Entity },
    #[error("node {node:?} is missing a TopologyPointers component")]
    MissingTopology { node: Entity },
    #[error("node {node:?} references a missing neighbor entity {neighbor:?}")]
    MissingNeighbor { node: Entity, neighbor: Entity },
}

/// Execute one local update sweep over all nodes.
///
/// This is *not* a standard dense forward pass. The system walks entities one at a time,
/// gathers only graph-local signals, and commits each node's activation in-place. That
/// in-place write is deliberate: it creates an asynchronous propagation surface where the
/// effective state seen by later nodes depends on the ordering chosen by the scheduler.
pub fn update_nodes_forward_forward(
    world: &mut World,
    activation_kind: ActivationKind,
) -> Result<(), NodeUpdateError> {
    let update_order: Vec<Entity> = world
        .query::<&TopologyPointers>()
        .iter()
        .map(|(entity, _)| entity)
        .collect();

    for entity in update_order {
        let neighbors = {
            let topology = world
                .get::<&TopologyPointers>(entity)
                .map_err(|_| NodeUpdateError::MissingTopology { node: entity })?;
            topology.neighbors
        };

        let neighbor_weights = {
            let weights = world
                .get::<&LocalWeights>(entity)
                .map_err(|_| NodeUpdateError::MissingWeights { node: entity })?;
            weights.neighbor_weights
        };

        let mut local_input_sum = 0.0f32;

        // The node never inspects any global tensor. It only accumulates from the three
        // entities named in its own topology component.
        for (slot, neighbor) in neighbors.iter().enumerate() {
            let neighbor_activation = {
                let neighbor_state = world.get::<&NodeState>(*neighbor).map_err(|_| {
                    NodeUpdateError::MissingNeighbor {
                        node: entity,
                        neighbor: *neighbor,
                    }
                })?;
                neighbor_state.activation
            };

            local_input_sum += neighbor_activation * neighbor_weights[slot];
        }

        let new_activation = activation_kind.apply(local_input_sum);

        {
            let mut state = world
                .get::<&mut NodeState>(entity)
                .map_err(|_| NodeUpdateError::MissingState { node: entity })?;
            state.activation = new_activation;
        }
    }

    Ok(())
}
