//! ECS-native contrastive data injection.
//!
//! Scheduling expectations:
//! - `inject_data_system` should run at the start of each simulation tick,
//! - then `update_nodes_forward_forward` can propagate the just-injected sample through
//!   the sparse graph,
//! - any learning rule that scores positive vs. negative activations should execute after
//!   the forward sweep completes.
//!
//! The current node state is scalar, so a 100-node graph cannot hold all 784 MNIST pixels
//! losslessly in a single tick. This system therefore injects one pixel per node when
//! there are 784 tagged input entities, and otherwise projects contiguous spans of the
//! flattened image onto the available input nodes by deterministic averaging.

use hecs::{Entity, World};
use thiserror::Error;

use crate::data::mnist_loader::{MnistDataset, MnistSampleError, MNIST_IMAGE_PIXELS};
use crate::data::procedural_negatives::{
    generate_hybrid_negative_from_indices, HybridNegativeError,
};
use crate::ecs_runtime::components::{InputNode, NodeState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimulationPhase {
    Positive,
    Negative,
}

impl SimulationPhase {
    pub fn toggled(self) -> Self {
        match self {
            Self::Positive => Self::Negative,
            Self::Negative => Self::Positive,
        }
    }

    pub fn learning_direction(self) -> f32 {
        match self {
            Self::Positive => 1.0,
            Self::Negative => -1.0,
        }
    }
}

#[derive(Debug)]
pub struct ContrastiveDataStream {
    dataset: MnistDataset,
    positive_cursor: usize,
    negative_cursor: usize,
}

impl ContrastiveDataStream {
    pub fn new(dataset: MnistDataset) -> Result<Self, IngestionError> {
        if dataset.is_empty() {
            return Err(IngestionError::EmptyDataset);
        }

        Ok(Self {
            dataset,
            positive_cursor: 0,
            negative_cursor: 0,
        })
    }

    pub fn dataset(&self) -> &MnistDataset {
        &self.dataset
    }

    pub fn fill_next_positive(
        &mut self,
        output: &mut [f32; MNIST_IMAGE_PIXELS],
    ) -> Result<PositiveSampleMetadata, IngestionError> {
        let index = self.positive_cursor % self.dataset.len();
        let label = self.dataset.fill_normalized_image(index, output)?;
        self.positive_cursor = (self.positive_cursor + 1) % self.dataset.len();

        Ok(PositiveSampleMetadata { index, label })
    }

    pub fn fill_next_negative(
        &mut self,
        output: &mut [f32; MNIST_IMAGE_PIXELS],
    ) -> Result<NegativeSampleMetadata, IngestionError> {
        let (upper_index, lower_index) = self.next_negative_pair()?;
        let hybrid =
            generate_hybrid_negative_from_indices(&self.dataset, upper_index, lower_index)?;

        output.copy_from_slice(&hybrid.pixels);
        self.negative_cursor = (self.negative_cursor + 1) % self.dataset.len();

        Ok(NegativeSampleMetadata {
            upper_index: hybrid.upper_index,
            lower_index: hybrid.lower_index,
            upper_label: hybrid.upper_label,
            lower_label: hybrid.lower_label,
        })
    }

    fn next_negative_pair(&self) -> Result<(usize, usize), IngestionError> {
        let len = self.dataset.len();
        if len < 2 {
            return Err(IngestionError::NotEnoughSamples { len });
        }

        let upper_index = self.negative_cursor % len;
        let upper_label = self.dataset.label(upper_index)?;

        let mut candidate = (upper_index + (len / 2).max(1)) % len;
        if candidate == upper_index {
            candidate = (candidate + 1) % len;
        }

        for _ in 0..len {
            if candidate != upper_index && self.dataset.label(candidate)? != upper_label {
                return Ok((upper_index, candidate));
            }
            candidate = (candidate + 1) % len;
        }

        Err(IngestionError::NoDistinctNegativePair {
            source_index: upper_index,
            source_label: upper_label,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositiveSampleMetadata {
    pub index: usize,
    pub label: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NegativeSampleMetadata {
    pub upper_index: usize,
    pub lower_index: usize,
    pub upper_label: u8,
    pub lower_label: u8,
}

pub fn ingestion_system(
    world: &mut World,
    stream: &mut ContrastiveDataStream,
    phase: &mut SimulationPhase,
) -> Result<(), IngestionError> {
    inject_data_system(world, stream, phase)
}

pub fn inject_data_system(
    world: &mut World,
    stream: &mut ContrastiveDataStream,
    phase: &mut SimulationPhase,
) -> Result<(), IngestionError> {
    let mut input_entities: Vec<Entity> = world
        .query::<&InputNode>()
        .iter()
        .map(|(entity, _)| entity)
        .collect();

    if input_entities.is_empty() {
        return Err(IngestionError::NoInputNodes);
    }

    input_entities.sort_unstable_by_key(|entity| entity.id());

    let mut pixels = [0.0f32; MNIST_IMAGE_PIXELS];
    match *phase {
        SimulationPhase::Positive => {
            let _ = stream.fill_next_positive(&mut pixels)?;
        }
        SimulationPhase::Negative => {
            let _ = stream.fill_next_negative(&mut pixels)?;
        }
    }

    write_pixels_into_inputs(world, &input_entities, &pixels)?;
    *phase = phase.toggled();

    Ok(())
}

#[derive(Debug, Error)]
pub enum IngestionError {
    #[error("the MNIST dataset is empty")]
    EmptyDataset,
    #[error("at least two MNIST samples are required for hybrid negative generation; found {len}")]
    NotEnoughSamples { len: usize },
    #[error("the world does not contain any entities tagged with InputNode")]
    NoInputNodes,
    #[error("input entity {node:?} is missing a NodeState component")]
    MissingState { node: Entity },
    #[error("failed to access an MNIST sample: {0}")]
    Sample(#[from] MnistSampleError),
    #[error("failed to build a hybrid negative: {0}")]
    Hybrid(#[from] HybridNegativeError),
    #[error("no distinct negative pair could be found for index {source_index} with label {source_label}")]
    NoDistinctNegativePair {
        source_index: usize,
        source_label: u8,
    },
}

fn write_pixels_into_inputs(
    world: &mut World,
    input_entities: &[Entity],
    pixels: &[f32; MNIST_IMAGE_PIXELS],
) -> Result<(), IngestionError> {
    for (slot, entity) in input_entities.iter().copied().enumerate() {
        let activation = projected_activation(pixels, slot, input_entities.len());
        let mut state = world
            .get::<&mut NodeState>(entity)
            .map_err(|_| IngestionError::MissingState { node: entity })?;
        state.activation = activation;
    }

    Ok(())
}

fn projected_activation(
    pixels: &[f32; MNIST_IMAGE_PIXELS],
    slot: usize,
    input_count: usize,
) -> f32 {
    if input_count >= MNIST_IMAGE_PIXELS {
        return pixels.get(slot).copied().unwrap_or(0.0);
    }

    let start = slot * MNIST_IMAGE_PIXELS / input_count;
    let mut end = (slot + 1) * MNIST_IMAGE_PIXELS / input_count;
    if end <= start {
        end = start + 1;
    }

    let span = &pixels[start..end];
    let sum: f32 = span.iter().copied().sum();
    sum / span.len() as f32
}

#[cfg(test)]
mod tests {
    use hecs::World;

    use super::*;
    use crate::data::mnist_loader::{normalize_mnist_byte, MnistDataset, MNIST_COLS, MNIST_ROWS};
    use crate::ecs_runtime::components::NodeState;

    #[test]
    fn inject_data_system_projects_positive_and_negative_samples_and_toggles_phase() {
        const EPSILON: f32 = 1.0e-5;

        let dataset = MnistDataset::from_raw_bytes(
            build_image_idx(&[[255u8; MNIST_IMAGE_PIXELS], [0u8; MNIST_IMAGE_PIXELS]])
                .into_boxed_slice(),
            build_label_idx(&[1, 8]).into_boxed_slice(),
        )
        .expect("construct dataset");

        let mut stream = ContrastiveDataStream::new(dataset).expect("stream");
        let mut world = World::new();
        let first = world.spawn((InputNode, NodeState::new(-1.0)));
        let second = world.spawn((InputNode, NodeState::new(-1.0)));
        let third = world.spawn((InputNode, NodeState::new(-1.0)));
        let fourth = world.spawn((InputNode, NodeState::new(-1.0)));
        let mut phase = SimulationPhase::Positive;

        inject_data_system(&mut world, &mut stream, &mut phase).expect("positive inject");
        assert_eq!(phase, SimulationPhase::Negative);
        assert!(
            (world.get::<&NodeState>(first).expect("first").activation - normalize_mnist_byte(255))
                .abs()
                < EPSILON
        );
        assert!(
            (world.get::<&NodeState>(fourth).expect("fourth").activation
                - normalize_mnist_byte(255))
            .abs()
                < EPSILON
        );

        inject_data_system(&mut world, &mut stream, &mut phase).expect("negative inject");
        assert_eq!(phase, SimulationPhase::Positive);
        assert!(
            (world
                .get::<&NodeState>(first)
                .expect("first negative")
                .activation
                - normalize_mnist_byte(255))
            .abs()
                < EPSILON
        );
        assert!(
            (world
                .get::<&NodeState>(fourth)
                .expect("fourth negative")
                .activation
                - normalize_mnist_byte(0))
            .abs()
                < EPSILON
        );

        let _ = (second, third);
    }

    fn build_image_idx(images: &[[u8; MNIST_IMAGE_PIXELS]]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16 + images.len() * MNIST_IMAGE_PIXELS);
        bytes.extend_from_slice(&2_051u32.to_be_bytes());
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
        bytes.extend_from_slice(&2_049u32.to_be_bytes());
        bytes.extend_from_slice(&(labels.len() as u32).to_be_bytes());
        bytes.extend_from_slice(labels);
        bytes
    }
}
