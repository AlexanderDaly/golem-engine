//! Experiment-level goodness metrics built on the same local score used for learning.

use hecs::{Entity, World};
use thiserror::Error;

use crate::ecs_runtime::components::{NodeState, TopologyPointers};
use crate::ecs_runtime::ingestion_system::SimulationPhase;
use crate::REGULAR_DEGREE;

/// Shared node-local goodness used for both learning and experiment evaluation.
///
/// `goodness = self_activation^2 + Σ(neighbor_activation^2)`
pub fn node_local_goodness(
    local_activation: f32,
    neighbor_activations: &[f32; REGULAR_DEGREE],
) -> f32 {
    local_activation * local_activation
        + neighbor_activations
            .iter()
            .map(|activation| activation * activation)
            .sum::<f32>()
}

#[derive(Debug, Error)]
pub enum ExperimentMetricError {
    #[error("node {node:?} is missing a NodeState component")]
    MissingState { node: Entity },
    #[error("node {node:?} is missing a TopologyPointers component")]
    MissingTopology { node: Entity },
    #[error("node {node:?} references a missing neighbor entity {neighbor:?}")]
    MissingNeighbor { node: Entity, neighbor: Entity },
    #[error("the world does not contain any graph nodes with TopologyPointers")]
    EmptyWorld,
    #[error("experiment goodness is missing positive observations")]
    MissingPositiveSamples,
    #[error("experiment goodness is missing negative observations")]
    MissingNegativeSamples,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExperimentGoodness {
    pub positive_mean_world_goodness: f32,
    pub negative_mean_world_goodness: f32,
    pub goodness_separation: f32,
    pub positive_samples: usize,
    pub negative_samples: usize,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ExperimentGoodnessAccumulator {
    positive_goodness_sum: f32,
    negative_goodness_sum: f32,
    positive_samples: usize,
    negative_samples: usize,
}

impl ExperimentGoodnessAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observe_world(
        &mut self,
        world: &World,
        phase: SimulationPhase,
    ) -> Result<(), ExperimentMetricError> {
        let mean_goodness = mean_world_goodness(world)?;

        match phase {
            SimulationPhase::Positive => {
                self.positive_goodness_sum += mean_goodness;
                self.positive_samples += 1;
            }
            SimulationPhase::Negative => {
                self.negative_goodness_sum += mean_goodness;
                self.negative_samples += 1;
            }
        }

        Ok(())
    }

    pub fn finish(&self) -> Result<ExperimentGoodness, ExperimentMetricError> {
        if self.positive_samples == 0 {
            return Err(ExperimentMetricError::MissingPositiveSamples);
        }
        if self.negative_samples == 0 {
            return Err(ExperimentMetricError::MissingNegativeSamples);
        }

        let positive_mean_world_goodness =
            self.positive_goodness_sum / self.positive_samples as f32;
        let negative_mean_world_goodness =
            self.negative_goodness_sum / self.negative_samples as f32;

        Ok(ExperimentGoodness {
            positive_mean_world_goodness,
            negative_mean_world_goodness,
            goodness_separation: positive_mean_world_goodness - negative_mean_world_goodness,
            positive_samples: self.positive_samples,
            negative_samples: self.negative_samples,
        })
    }
}

pub fn mean_world_goodness(world: &World) -> Result<f32, ExperimentMetricError> {
    let update_order: Vec<Entity> = world
        .query::<&TopologyPointers>()
        .iter()
        .map(|(entity, _)| entity)
        .collect();
    let node_count = update_order.len();

    if node_count == 0 {
        return Err(ExperimentMetricError::EmptyWorld);
    }

    let mut goodness_sum = 0.0f32;

    for entity in update_order {
        let neighbors = node_neighbors(world, entity)?;
        let local_activation = node_activation(world, entity)?;
        let neighbor_activations = neighbor_activations(world, entity, &neighbors)?;
        goodness_sum += node_local_goodness(local_activation, &neighbor_activations);
    }

    Ok(goodness_sum / node_count as f32)
}

fn node_neighbors(
    world: &World,
    entity: Entity,
) -> Result<[Entity; REGULAR_DEGREE], ExperimentMetricError> {
    let topology = world
        .get::<&TopologyPointers>(entity)
        .map_err(|_| ExperimentMetricError::MissingTopology { node: entity })?;
    Ok(topology.neighbors)
}

fn node_activation(world: &World, entity: Entity) -> Result<f32, ExperimentMetricError> {
    let state = world
        .get::<&NodeState>(entity)
        .map_err(|_| ExperimentMetricError::MissingState { node: entity })?;
    Ok(state.activation)
}

fn neighbor_activations(
    world: &World,
    entity: Entity,
    neighbors: &[Entity; REGULAR_DEGREE],
) -> Result<[f32; REGULAR_DEGREE], ExperimentMetricError> {
    let mut activations = [0.0; REGULAR_DEGREE];

    for (slot, neighbor) in neighbors.iter().copied().enumerate() {
        activations[slot] = world
            .get::<&NodeState>(neighbor)
            .map_err(|_| ExperimentMetricError::MissingNeighbor {
                node: entity,
                neighbor,
            })?
            .activation;
    }

    Ok(activations)
}

#[cfg(test)]
mod tests {
    use hecs::World;

    use super::*;

    const EPSILON: f32 = 1.0e-6;

    #[test]
    fn node_local_goodness_matches_learning_formula() {
        let goodness = node_local_goodness(0.8, &[1.0, -0.5, 0.25]);

        assert!((goodness - 1.9525).abs() < EPSILON);
    }

    #[test]
    fn mean_world_goodness_averages_node_local_scores() {
        let world = complete_four_node_world([1.0, -0.5, 0.25, -0.75]);

        let mean_goodness = mean_world_goodness(&world).expect("mean world goodness");

        assert!((mean_goodness - 1.875).abs() < EPSILON);
    }

    #[test]
    fn accumulator_buckets_phases_and_computes_separation() {
        let mut accumulator = ExperimentGoodnessAccumulator::new();
        let positive_a = complete_four_node_world([1.0, 0.0, 0.0, 0.0]);
        let positive_b = complete_four_node_world([1.0, 1.0, 0.0, 0.0]);
        let negative = complete_four_node_world([0.5, 0.5, 0.0, 0.0]);

        accumulator
            .observe_world(&positive_a, SimulationPhase::Positive)
            .expect("first positive observation");
        accumulator
            .observe_world(&positive_b, SimulationPhase::Positive)
            .expect("second positive observation");
        accumulator
            .observe_world(&negative, SimulationPhase::Negative)
            .expect("negative observation");

        let goodness = accumulator.finish().expect("completed experiment goodness");

        assert_eq!(goodness.positive_samples, 2);
        assert_eq!(goodness.negative_samples, 1);
        assert!((goodness.positive_mean_world_goodness - 1.5).abs() < EPSILON);
        assert!((goodness.negative_mean_world_goodness - 0.5).abs() < EPSILON);
        assert!((goodness.goodness_separation - 1.0).abs() < EPSILON);
    }

    #[test]
    fn accumulator_requires_both_phases() {
        let mut accumulator = ExperimentGoodnessAccumulator::new();
        let positive = complete_four_node_world([1.0, 0.0, 0.0, 0.0]);

        accumulator
            .observe_world(&positive, SimulationPhase::Positive)
            .expect("positive observation");

        let error = accumulator
            .finish()
            .expect_err("missing negative observations should fail");

        assert!(matches!(
            error,
            ExperimentMetricError::MissingNegativeSamples
        ));
    }

    fn complete_four_node_world(activations: [f32; 4]) -> World {
        use crate::ecs_runtime::components::{NodeState, TopologyPointers};

        let mut world = World::new();
        let entities: [Entity; 4] =
            std::array::from_fn(|index| world.spawn((NodeState::new(activations[index]),)));

        for (index, entity) in entities.iter().copied().enumerate() {
            let neighbors = std::array::from_fn(|slot| entities[(index + slot + 1) % 4]);
            world
                .insert(entity, (TopologyPointers::new(neighbors),))
                .expect("attach topology");
        }

        world
    }
}
