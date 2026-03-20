//! ECS systems for the forward-forward style local dynamics.
//!
//! The central architectural rule for this file is that updates remain node-local:
//! - each entity reads only the activations stored on its own topological neighbors,
//! - each entity applies only its own local edge weights,
//! - each entity writes back only its own activation,
//! - writes happen immediately, so later entities in the same tick can observe earlier
//!   updates and there is no hidden global synchronization barrier.

use std::fmt;
use std::str::FromStr;

use hecs::{Entity, World};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ecs_runtime::components::{LocalWeights, NodeState, TopologyPointers};
use crate::ecs_runtime::ingestion_system::SimulationPhase;
use crate::REGULAR_DEGREE;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationKind {
    Tanh,
    Relu,
    SoftSign,
}

impl ActivationKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tanh => "tanh",
            Self::Relu => "relu",
            Self::SoftSign => "softsign",
        }
    }

    pub fn apply(self, input: f32) -> f32 {
        match self {
            Self::Tanh => input.tanh(),
            Self::Relu => input.max(0.0),
            Self::SoftSign => input / (1.0 + input.abs()),
        }
    }
}

impl fmt::Display for ActivationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ActivationKind {
    type Err = ParseActivationKindError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "tanh" => Ok(Self::Tanh),
            "relu" => Ok(Self::Relu),
            "softsign" => Ok(Self::SoftSign),
            _ => Err(ParseActivationKindError {
                value: value.to_owned(),
            }),
        }
    }
}

#[derive(Debug, Error)]
#[error("unsupported activation kind {value:?}; expected one of: tanh, relu, softsign")]
pub struct ParseActivationKindError {
    pub value: String,
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
    let update_order = collect_update_order(world);

    for entity in update_order {
        let neighbors = node_neighbors(world, entity)?;
        let neighbor_weights = node_weights(world, entity)?;
        let neighbor_activations = neighbor_activations(world, entity, &neighbors)?;

        // The node never inspects any global tensor. It only accumulates from the three
        // entities named in its own topology component.
        let local_input_sum: f32 = neighbor_weights
            .iter()
            .zip(neighbor_activations.iter())
            .map(|(weight, activation)| weight * activation)
            .sum();

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

/// Apply one local Forward-Forward learning sweep after a forward pass has completed.
///
/// Each node computes a goodness score from the activations visible in its local
/// neighborhood:
///
/// `goodness = self_activation^2 + Σ(neighbor_activation^2)`
///
/// It then updates only its own incoming edge weights using the same local signals:
///
/// `Δw_i = phase_direction * learning_rate * goodness * self_activation * neighbor_activation_i`
///
/// Positive phases therefore reinforce locally correlated activations, while negative
/// phases suppress them. Pass the phase associated with the sample that just completed
/// the forward sweep. If this runs immediately after `inject_data_system`, that means
/// using the pre-toggle phase value, or equivalently `phase.toggled()` on the
/// post-ingestion scheduler state.
pub fn update_local_weights_forward_forward(
    world: &mut World,
    phase: SimulationPhase,
    learning_rate: f32,
) -> Result<(), NodeUpdateError> {
    let update_order = collect_update_order(world);
    let phase_direction = phase.learning_direction();

    for entity in update_order {
        let neighbors = node_neighbors(world, entity)?;
        let local_activation = node_activation(world, entity)?;
        let neighbor_activations = neighbor_activations(world, entity, &neighbors)?;
        let goodness = local_goodness(local_activation, &neighbor_activations);

        let mut weights = world
            .get::<&mut LocalWeights>(entity)
            .map_err(|_| NodeUpdateError::MissingWeights { node: entity })?;

        for (slot, neighbor_activation) in neighbor_activations.iter().copied().enumerate() {
            let local_correlation = local_activation * neighbor_activation;
            let delta = phase_direction * learning_rate * goodness * local_correlation;
            weights.neighbor_weights[slot] += delta;
        }
    }

    Ok(())
}

fn collect_update_order(world: &World) -> Vec<Entity> {
    world
        .query::<&TopologyPointers>()
        .iter()
        .map(|(entity, _)| entity)
        .collect()
}

fn node_neighbors(
    world: &World,
    entity: Entity,
) -> Result<[Entity; REGULAR_DEGREE], NodeUpdateError> {
    let topology = world
        .get::<&TopologyPointers>(entity)
        .map_err(|_| NodeUpdateError::MissingTopology { node: entity })?;
    Ok(topology.neighbors)
}

fn node_weights(world: &World, entity: Entity) -> Result<[f32; REGULAR_DEGREE], NodeUpdateError> {
    let weights = world
        .get::<&LocalWeights>(entity)
        .map_err(|_| NodeUpdateError::MissingWeights { node: entity })?;
    Ok(weights.neighbor_weights)
}

fn node_activation(world: &World, entity: Entity) -> Result<f32, NodeUpdateError> {
    let state = world
        .get::<&NodeState>(entity)
        .map_err(|_| NodeUpdateError::MissingState { node: entity })?;
    Ok(state.activation)
}

fn neighbor_activations(
    world: &World,
    entity: Entity,
    neighbors: &[Entity; REGULAR_DEGREE],
) -> Result<[f32; REGULAR_DEGREE], NodeUpdateError> {
    let mut activations = [0.0; REGULAR_DEGREE];

    for (slot, neighbor) in neighbors.iter().copied().enumerate() {
        activations[slot] = world
            .get::<&NodeState>(neighbor)
            .map_err(|_| NodeUpdateError::MissingNeighbor {
                node: entity,
                neighbor,
            })?
            .activation;
    }

    Ok(activations)
}

fn local_goodness(local_activation: f32, neighbor_activations: &[f32; REGULAR_DEGREE]) -> f32 {
    local_activation * local_activation
        + neighbor_activations
            .iter()
            .map(|activation| activation * activation)
            .sum::<f32>()
}

#[cfg(test)]
mod tests {
    use hecs::World;

    use super::*;
    use crate::ecs_runtime::components::{LocalWeights, NodeState, TopologyPointers};

    const EPSILON: f32 = 1.0e-6;

    #[test]
    fn update_local_weights_forward_forward_reinforces_goodness_during_positive_phase() {
        let (mut world, node) = single_learning_node_world();

        update_local_weights_forward_forward(&mut world, SimulationPhase::Positive, 0.1)
            .expect("positive learning update");

        let weights = world
            .get::<&LocalWeights>(node)
            .expect("weights after positive update");

        assert!((weights.neighbor_weights[0] - 0.6562).abs() < EPSILON);
        assert!((weights.neighbor_weights[1] + 0.3281).abs() < EPSILON);
        assert!((weights.neighbor_weights[2] - 0.13905).abs() < EPSILON);
    }

    #[test]
    fn update_local_weights_forward_forward_suppresses_goodness_during_negative_phase() {
        let (mut world, node) = single_learning_node_world();

        update_local_weights_forward_forward(&mut world, SimulationPhase::Negative, 0.1)
            .expect("negative learning update");

        let weights = world
            .get::<&LocalWeights>(node)
            .expect("weights after negative update");

        assert!((weights.neighbor_weights[0] - 0.3438).abs() < EPSILON);
        assert!((weights.neighbor_weights[1] + 0.1719).abs() < EPSILON);
        assert!((weights.neighbor_weights[2] - 0.06095).abs() < EPSILON);
    }

    fn single_learning_node_world() -> (World, Entity) {
        let mut world = World::new();
        let neighbor_a = world.spawn((NodeState::new(1.0),));
        let neighbor_b = world.spawn((NodeState::new(-0.5),));
        let neighbor_c = world.spawn((NodeState::new(0.25),));
        let node = world.spawn((
            NodeState::new(0.8),
            LocalWeights::new([0.5, -0.25, 0.1]),
            TopologyPointers::new([neighbor_a, neighbor_b, neighbor_c]),
        ));

        (world, node)
    }
}
