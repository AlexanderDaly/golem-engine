//! Forward-Forward evaluation helpers.
//!
//! Evaluation deliberately operates on an isolated clone of the training world so it can
//! mutate activations while guaranteeing that no live training weights are touched.
//!
//! World-level goodness is defined as the mean squared activation over all graph nodes after
//! a single forward sweep from a freshly reset activation state. This keeps the metric
//! independent of graph size and avoids counting each graph edge multiple times.

use hecs::World;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::data::mnist_loader::{MnistDataset, MNIST_IMAGE_PIXELS};
use crate::ecs_runtime::checkpoint::GraphCheckpoint;
use crate::ecs_runtime::components::NodeState;
use crate::ecs_runtime::ingestion_system::{
    fill_positive_sample, inject_conditioned_sample, IngestionError, SimulationPhase,
};
use crate::ecs_runtime::systems::{update_nodes_forward_forward, ActivationKind, NodeUpdateError};
use crate::MNIST_LABEL_CLASS_COUNT;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvaluationSummary {
    pub accuracy: f32,
    pub mean_correct_goodness: f32,
    pub mean_best_wrong_goodness: f32,
    pub mean_margin: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub(crate) struct EvaluationAccumulator {
    pub sample_count: usize,
    pub correct_predictions: usize,
    pub correct_goodness_sum: f32,
    pub best_wrong_goodness_sum: f32,
    pub margin_sum: f32,
}

impl EvaluationAccumulator {
    fn record(&mut self, evaluation: SampleEvaluation, true_label: u8) {
        self.sample_count += 1;
        self.correct_predictions += usize::from(evaluation.predicted_label == true_label);
        self.correct_goodness_sum += evaluation.correct_goodness;
        self.best_wrong_goodness_sum += evaluation.best_wrong_goodness;
        self.margin_sum += evaluation.margin();
    }

    pub fn finish(self) -> EvaluationSummary {
        let sample_count = self.sample_count as f32;
        EvaluationSummary {
            accuracy: self.correct_predictions as f32 / sample_count,
            mean_correct_goodness: self.correct_goodness_sum / sample_count,
            mean_best_wrong_goodness: self.best_wrong_goodness_sum / sample_count,
            mean_margin: self.margin_sum / sample_count,
        }
    }
}

#[derive(Debug, Error)]
pub enum EvaluationError {
    #[error("checkpoint operation failed while preparing the evaluation world: {0}")]
    Checkpoint(#[from] crate::ecs_runtime::checkpoint::CheckpointError),
    #[error("failed to evaluate dataset samples: {0}")]
    Ingestion(#[from] IngestionError),
    #[error("evaluation forward sweep failed: {0}")]
    NodeUpdate(#[from] NodeUpdateError),
}

pub fn evaluate_forward_forward(
    world: &World,
    phase: SimulationPhase,
    dataset: &MnistDataset,
    activation_kind: ActivationKind,
) -> Result<EvaluationSummary, EvaluationError> {
    Ok(
        evaluate_forward_forward_range(world, phase, dataset, activation_kind, 0..dataset.len())?
            .finish(),
    )
}

pub(crate) fn evaluate_forward_forward_range(
    world: &World,
    phase: SimulationPhase,
    dataset: &MnistDataset,
    activation_kind: ActivationKind,
    range: std::ops::Range<usize>,
) -> Result<EvaluationAccumulator, EvaluationError> {
    let (mut eval_world, _) = GraphCheckpoint::from_world(world, phase)?.into_world()?;
    let mut pixels = [0.0f32; MNIST_IMAGE_PIXELS];
    let mut accumulator = EvaluationAccumulator::default();

    for index in range {
        let sample = fill_positive_sample(dataset, index, &mut pixels)?;
        let evaluation = evaluate_conditioned_pixels(
            &mut eval_world,
            &pixels,
            sample.true_label,
            activation_kind,
        )?;

        accumulator.record(evaluation, sample.true_label);
    }

    Ok(accumulator)
}

pub fn world_goodness(world: &World) -> f32 {
    let mut activation_count = 0usize;
    let mut squared_activation_sum = 0.0f32;

    for (_entity, state) in world.query::<&NodeState>().iter() {
        activation_count += 1;
        squared_activation_sum += state.activation * state.activation;
    }

    if activation_count == 0 {
        0.0
    } else {
        squared_activation_sum / activation_count as f32
    }
}

fn evaluate_conditioned_pixels(
    world: &mut World,
    pixels: &[f32; MNIST_IMAGE_PIXELS],
    true_label: u8,
    activation_kind: ActivationKind,
) -> Result<SampleEvaluation, EvaluationError> {
    let mut predicted_label = 0u8;
    let mut predicted_goodness = f32::NEG_INFINITY;
    let mut correct_goodness = f32::NEG_INFINITY;
    let mut best_wrong_goodness = f32::NEG_INFINITY;

    for candidate_label in 0..MNIST_LABEL_CLASS_COUNT as u8 {
        reset_activations(world);
        inject_conditioned_sample(world, pixels, candidate_label)?;
        update_nodes_forward_forward(world, activation_kind)?;
        let goodness = world_goodness(world);

        if goodness > predicted_goodness {
            predicted_label = candidate_label;
            predicted_goodness = goodness;
        }

        if candidate_label == true_label {
            correct_goodness = goodness;
        } else if goodness > best_wrong_goodness {
            best_wrong_goodness = goodness;
        }
    }

    Ok(SampleEvaluation {
        predicted_label,
        correct_goodness,
        best_wrong_goodness,
    })
}

fn reset_activations(world: &mut World) {
    let mut query = world.query::<&mut NodeState>();
    for (_entity, state) in query.iter() {
        state.activation = 0.0;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SampleEvaluation {
    predicted_label: u8,
    correct_goodness: f32,
    best_wrong_goodness: f32,
}

impl SampleEvaluation {
    fn margin(self) -> f32 {
        self.correct_goodness - self.best_wrong_goodness
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::mnist_loader::{
        normalize_mnist_byte, MnistDataset, MNIST_COLS, MNIST_IMAGE_MAGIC, MNIST_IMAGE_PIXELS,
        MNIST_LABEL_MAGIC, MNIST_ROWS,
    };
    use crate::ecs_runtime::components::{
        InputNode, LocalWeights, NodeState, StableNodeIndex, TopologyPointers,
    };
    use crate::CONDITIONED_INPUT_NODE_COUNT;

    const EPSILON: f32 = 1.0e-6;

    #[test]
    fn evaluation_predicts_the_true_label_on_a_synthetic_conditioned_world() {
        let dataset = MnistDataset::from_raw_bytes(
            build_image_idx(&[sample_with_hot_pixel(1), sample_with_hot_pixel(8)])
                .into_boxed_slice(),
            build_label_idx(&[1u8, 8u8]).into_boxed_slice(),
        )
        .expect("dataset");
        let world = classification_world();

        let summary = evaluate_forward_forward(
            &world,
            SimulationPhase::Positive,
            &dataset,
            ActivationKind::Relu,
        )
        .expect("evaluation summary");

        let active_pixel = normalize_mnist_byte(255);
        let correct_goodness =
            (active_pixel * active_pixel + 1.0 + (active_pixel + 1.0) * (active_pixel + 1.0))
                / (CONDITIONED_INPUT_NODE_COUNT + MNIST_LABEL_CLASS_COUNT) as f32;
        let best_wrong_goodness =
            (active_pixel * active_pixel + 1.0 + active_pixel * active_pixel + 1.0)
                / (CONDITIONED_INPUT_NODE_COUNT + MNIST_LABEL_CLASS_COUNT) as f32;

        assert!((summary.accuracy - 1.0).abs() < EPSILON);
        assert!((summary.mean_correct_goodness - correct_goodness).abs() < EPSILON);
        assert!((summary.mean_best_wrong_goodness - best_wrong_goodness).abs() < EPSILON);
        assert!((summary.mean_margin - (correct_goodness - best_wrong_goodness)).abs() < EPSILON);
    }

    #[test]
    fn world_goodness_averages_squared_activation() {
        let mut world = World::new();
        let a = world.spawn((NodeState::new(3.0),));
        let _b = world.spawn((NodeState::new(4.0),));

        assert_eq!(world_goodness(&world), 12.5);
        assert_eq!(
            world.get::<&NodeState>(a).expect("node state").activation,
            3.0
        );
    }

    fn classification_world() -> World {
        let mut world = World::new();
        let input_entities: Vec<_> = (0..CONDITIONED_INPUT_NODE_COUNT)
            .map(|index| {
                world.spawn((
                    InputNode,
                    StableNodeIndex::new(index),
                    NodeState::new(0.0),
                    LocalWeights::new([1.0 / 3.0; 3]),
                ))
            })
            .collect();
        let classifier_entities: Vec<_> = (0..MNIST_LABEL_CLASS_COUNT)
            .map(|label| {
                world.spawn((
                    StableNodeIndex::new(CONDITIONED_INPUT_NODE_COUNT + label),
                    NodeState::new(0.0),
                    LocalWeights::new([1.0, 1.0, 0.0]),
                ))
            })
            .collect();

        for entity in input_entities.iter().copied() {
            world
                .insert(entity, (TopologyPointers::new([entity, entity, entity]),))
                .expect("attach input topology");
        }

        for (label, entity) in classifier_entities.iter().copied().enumerate() {
            let pixel = input_entities[label];
            let label_input = input_entities[MNIST_IMAGE_PIXELS + label];
            world
                .insert(
                    entity,
                    (TopologyPointers::new([pixel, label_input, entity]),),
                )
                .expect("attach classifier topology");
        }

        world
    }

    fn sample_with_hot_pixel(pixel_index: usize) -> [u8; MNIST_IMAGE_PIXELS] {
        let mut pixels = [0u8; MNIST_IMAGE_PIXELS];
        pixels[pixel_index] = 255;
        pixels
    }

    fn build_image_idx(images: &[[u8; MNIST_IMAGE_PIXELS]]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16 + images.len() * MNIST_IMAGE_PIXELS);
        bytes.extend_from_slice(&MNIST_IMAGE_MAGIC.to_be_bytes());
        bytes.extend_from_slice(&(images.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&(MNIST_ROWS as u32).to_be_bytes());
        bytes.extend_from_slice(&(MNIST_COLS as u32).to_be_bytes());
        for image in images {
            bytes.extend_from_slice(image);
        }
        bytes
    }

    fn build_label_idx(labels: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 + labels.len());
        bytes.extend_from_slice(&MNIST_LABEL_MAGIC.to_be_bytes());
        bytes.extend_from_slice(&(labels.len() as u32).to_be_bytes());
        bytes.extend_from_slice(labels);
        bytes
    }
}
