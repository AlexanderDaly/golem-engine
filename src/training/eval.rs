//! Forward-Forward evaluation helpers.
//!
//! Evaluation deliberately operates on an isolated clone of the training world so it can
//! mutate activations while guaranteeing that no live training weights are touched.
//!
//! World-level goodness is defined as the mean squared activation over all graph nodes after
//! a single forward sweep from a freshly reset activation state. This keeps the metric
//! independent of graph size and avoids counting each graph edge multiple times.

use hecs::World;
use thiserror::Error;

use crate::data::mnist_loader::{MnistDataset, MNIST_IMAGE_PIXELS};
use crate::ecs_runtime::checkpoint::GraphCheckpoint;
use crate::ecs_runtime::components::NodeState;
use crate::ecs_runtime::ingestion_system::{
    inject_pixels, ContrastiveDataStream, IngestionError, SimulationPhase,
};
use crate::ecs_runtime::systems::{update_nodes_forward_forward, ActivationKind, NodeUpdateError};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvaluationSummary {
    pub mean_positive_goodness: f32,
    pub mean_negative_goodness: f32,
    pub goodness_separation: f32,
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
    let (mut eval_world, _) = GraphCheckpoint::from_world(world, phase)?.into_world()?;
    let mut stream = ContrastiveDataStream::new(dataset)?;
    let sample_count = stream.dataset().len();
    let mut pixels = [0.0f32; MNIST_IMAGE_PIXELS];
    let mut positive_goodness_sum = 0.0f32;
    let mut negative_goodness_sum = 0.0f32;

    for _ in 0..sample_count {
        let _ = stream.fill_next_positive(&mut pixels)?;
        positive_goodness_sum += evaluate_pixels(&mut eval_world, &pixels, activation_kind)?;

        let _ = stream.fill_next_negative(&mut pixels)?;
        negative_goodness_sum += evaluate_pixels(&mut eval_world, &pixels, activation_kind)?;
    }

    let sample_count = sample_count as f32;
    let mean_positive_goodness = positive_goodness_sum / sample_count;
    let mean_negative_goodness = negative_goodness_sum / sample_count;

    Ok(EvaluationSummary {
        mean_positive_goodness,
        mean_negative_goodness,
        goodness_separation: mean_positive_goodness - mean_negative_goodness,
    })
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

fn evaluate_pixels(
    world: &mut World,
    pixels: &[f32; MNIST_IMAGE_PIXELS],
    activation_kind: ActivationKind,
) -> Result<f32, EvaluationError> {
    reset_activations(world);
    inject_pixels(world, pixels)?;
    update_nodes_forward_forward(world, activation_kind)?;
    Ok(world_goodness(world))
}

fn reset_activations(world: &mut World) {
    let mut query = world.query::<&mut NodeState>();
    for (_entity, state) in query.iter() {
        state.activation = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::mnist_loader::{
        MnistDataset, MNIST_COLS, MNIST_IMAGE_MAGIC, MNIST_IMAGE_PIXELS, MNIST_LABEL_MAGIC,
        MNIST_ROWS,
    };
    use crate::ecs_runtime::components::{
        InputNode, LocalWeights, NodeState, StableNodeIndex, TopologyPointers,
    };
    use crate::ecs_runtime::ingestion_system::{fill_negative_sample, fill_positive_sample};

    #[test]
    fn evaluation_prefers_positive_goodness_on_a_synthetic_world() {
        let dataset = MnistDataset::from_raw_bytes(
            build_image_idx(&[[255u8; MNIST_IMAGE_PIXELS], [0u8; MNIST_IMAGE_PIXELS]])
                .into_boxed_slice(),
            build_label_idx(&[1u8, 8u8]).into_boxed_slice(),
        )
        .expect("dataset");
        let (world, node) = single_node_world();

        let summary = evaluate_forward_forward(
            &world,
            SimulationPhase::Positive,
            &dataset,
            ActivationKind::SoftSign,
        )
        .expect("evaluation summary");

        assert!(summary.mean_positive_goodness > summary.mean_negative_goodness);
        assert!(summary.goodness_separation > 0.0);
        assert_eq!(
            world
                .get::<&NodeState>(node)
                .expect("node state")
                .activation,
            3.5
        );
    }

    #[test]
    fn helper_fill_functions_match_stream_ordering() {
        let dataset = MnistDataset::from_raw_bytes(
            build_image_idx(&[[255u8; MNIST_IMAGE_PIXELS], [0u8; MNIST_IMAGE_PIXELS]])
                .into_boxed_slice(),
            build_label_idx(&[1u8, 8u8]).into_boxed_slice(),
        )
        .expect("dataset");
        let mut positive = [0.0f32; MNIST_IMAGE_PIXELS];
        let mut negative = [0.0f32; MNIST_IMAGE_PIXELS];

        let positive_metadata =
            fill_positive_sample(&dataset, 0, &mut positive).expect("positive sample");
        let negative_metadata =
            fill_negative_sample(&dataset, 0, &mut negative).expect("negative sample");

        assert_eq!(positive_metadata.index, 0);
        assert_eq!(negative_metadata.upper_index, 0);
        assert_eq!(negative_metadata.lower_index, 1);
        assert_ne!(positive, negative);
    }

    fn single_node_world() -> (World, hecs::Entity) {
        let mut world = World::new();
        let node = world.spawn((
            InputNode,
            StableNodeIndex::new(0),
            NodeState::new(3.5),
            LocalWeights::new([0.1, 0.1, 0.1]),
        ));

        world
            .insert(node, (TopologyPointers::new([node, node, node]),))
            .expect("topology");
        (world, node)
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
