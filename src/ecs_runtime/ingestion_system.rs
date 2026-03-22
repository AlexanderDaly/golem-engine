//! ECS-native label-conditioned data injection.
//!
//! Scheduling expectations:
//! - `inject_data_system` should run at the start of each simulation tick,
//! - then `update_nodes_forward_forward` can propagate the just-injected sample through
//!   the sparse graph,
//! - any learning rule that scores positive vs. negative activations should execute after
//!   the forward sweep completes.
//!
//! Label-conditioned Forward-Forward uses a fixed conditioned input surface:
//! - input slots `0..783` carry normalized MNIST pixels,
//! - input slots `784..793` carry a one-hot candidate digit label,
//! - any additional input-tagged entities are zeroed so the first 794 slots define the
//!   entire conditioned sample.

use hecs::{Entity, World};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::data::mnist_loader::{MnistDataset, MnistSampleError, MNIST_IMAGE_PIXELS};
use crate::ecs_runtime::components::{InputNode, NodeState, StableNodeIndex};
use crate::{CONDITIONED_INPUT_NODE_COUNT, MNIST_LABEL_CLASS_COUNT};

const LABEL_SLOT_OFFSET: usize = MNIST_IMAGE_PIXELS;
const LABEL_ON_ACTIVATION: f32 = 1.0;
const LABEL_OFF_ACTIVATION: f32 = 0.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
pub struct ContrastiveDataStream<'a> {
    dataset: &'a MnistDataset,
    positive_cursor: usize,
    negative_cursor: usize,
}

impl<'a> ContrastiveDataStream<'a> {
    pub fn new(dataset: &'a MnistDataset) -> Result<Self, IngestionError> {
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
        self.dataset
    }

    pub fn fill_next_positive(
        &mut self,
        output: &mut [f32; MNIST_IMAGE_PIXELS],
    ) -> Result<ConditionedSampleMetadata, IngestionError> {
        let metadata = fill_positive_sample(self.dataset, self.positive_cursor, output)?;
        self.positive_cursor = (self.positive_cursor + 1) % self.dataset.len();

        Ok(metadata)
    }

    pub fn fill_next_negative(
        &mut self,
        output: &mut [f32; MNIST_IMAGE_PIXELS],
    ) -> Result<ConditionedSampleMetadata, IngestionError> {
        let metadata = fill_negative_sample(self.dataset, self.negative_cursor, output)?;
        self.negative_cursor = (self.negative_cursor + 1) % self.dataset.len();

        Ok(metadata)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConditionedSampleMetadata {
    pub index: usize,
    pub true_label: u8,
    pub candidate_label: u8,
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
    stream: &mut ContrastiveDataStream<'_>,
    phase: &mut SimulationPhase,
) -> Result<(), IngestionError> {
    let mut pixels = [0.0f32; MNIST_IMAGE_PIXELS];
    let metadata = match *phase {
        SimulationPhase::Positive => stream.fill_next_positive(&mut pixels)?,
        SimulationPhase::Negative => stream.fill_next_negative(&mut pixels)?,
    };

    inject_conditioned_sample(world, &pixels, metadata.candidate_label)?;
    *phase = phase.toggled();

    Ok(())
}

pub fn fill_positive_sample(
    dataset: &MnistDataset,
    cursor: usize,
    output: &mut [f32; MNIST_IMAGE_PIXELS],
) -> Result<ConditionedSampleMetadata, IngestionError> {
    if dataset.is_empty() {
        return Err(IngestionError::EmptyDataset);
    }

    let index = cursor % dataset.len();
    let true_label = dataset.fill_normalized_image(index, output)?;
    ensure_valid_digit_label(index, true_label)?;

    Ok(ConditionedSampleMetadata {
        index,
        true_label,
        candidate_label: true_label,
    })
}

pub fn fill_negative_sample(
    dataset: &MnistDataset,
    cursor: usize,
    output: &mut [f32; MNIST_IMAGE_PIXELS],
) -> Result<ConditionedSampleMetadata, IngestionError> {
    if dataset.is_empty() {
        return Err(IngestionError::EmptyDataset);
    }

    let index = cursor % dataset.len();
    let true_label = dataset.fill_normalized_image(index, output)?;
    ensure_valid_digit_label(index, true_label)?;
    let candidate_label = deterministic_wrong_label(true_label, index);

    Ok(ConditionedSampleMetadata {
        index,
        true_label,
        candidate_label,
    })
}

pub fn inject_conditioned_sample(
    world: &mut World,
    pixels: &[f32; MNIST_IMAGE_PIXELS],
    candidate_label: u8,
) -> Result<(), IngestionError> {
    ensure_valid_candidate_label(candidate_label)?;
    let input_entities = sorted_input_entities(world)?;
    write_conditioned_inputs(world, &input_entities, pixels, candidate_label)
}

#[derive(Debug, Error)]
pub enum IngestionError {
    #[error("the MNIST dataset is empty")]
    EmptyDataset,
    #[error(
        "label-conditioned Forward-Forward requires at least {required} InputNode entities; found {found}"
    )]
    InsufficientInputNodes { found: usize, required: usize },
    #[error("input entity {node:?} is missing a NodeState component")]
    MissingState { node: Entity },
    #[error("MNIST sample {index} carries label {label}, but only 0..={max_label} are supported")]
    InvalidDigitLabel {
        index: usize,
        label: u8,
        max_label: u8,
    },
    #[error("candidate label {label} is out of range; expected 0..={max_label}")]
    InvalidCandidateLabel { label: u8, max_label: u8 },
    #[error("failed to access an MNIST sample: {0}")]
    Sample(#[from] MnistSampleError),
}

fn write_conditioned_inputs(
    world: &mut World,
    input_entities: &[Entity],
    pixels: &[f32; MNIST_IMAGE_PIXELS],
    candidate_label: u8,
) -> Result<(), IngestionError> {
    for (slot, entity) in input_entities.iter().copied().enumerate() {
        let activation = conditioned_slot_activation(pixels, slot, candidate_label);
        let mut state = world
            .get::<&mut NodeState>(entity)
            .map_err(|_| IngestionError::MissingState { node: entity })?;
        state.activation = activation;
    }

    Ok(())
}

fn sorted_input_entities(world: &World) -> Result<Vec<Entity>, IngestionError> {
    let mut input_entities: Vec<Entity> = world
        .query::<&InputNode>()
        .iter()
        .map(|(entity, _)| entity)
        .collect();

    if input_entities.len() < CONDITIONED_INPUT_NODE_COUNT {
        return Err(IngestionError::InsufficientInputNodes {
            found: input_entities.len(),
            required: CONDITIONED_INPUT_NODE_COUNT,
        });
    }

    input_entities.sort_unstable_by_key(|entity| input_sort_key(world, *entity));
    Ok(input_entities)
}

fn input_sort_key(world: &World, entity: Entity) -> usize {
    world
        .get::<&StableNodeIndex>(entity)
        .map(|stable_index| stable_index.index)
        .unwrap_or_else(|_| entity.id() as usize)
}

fn ensure_valid_digit_label(index: usize, label: u8) -> Result<(), IngestionError> {
    if usize::from(label) >= MNIST_LABEL_CLASS_COUNT {
        return Err(IngestionError::InvalidDigitLabel {
            index,
            label,
            max_label: (MNIST_LABEL_CLASS_COUNT - 1) as u8,
        });
    }

    Ok(())
}

fn ensure_valid_candidate_label(label: u8) -> Result<(), IngestionError> {
    if usize::from(label) >= MNIST_LABEL_CLASS_COUNT {
        return Err(IngestionError::InvalidCandidateLabel {
            label,
            max_label: (MNIST_LABEL_CLASS_COUNT - 1) as u8,
        });
    }

    Ok(())
}

fn deterministic_wrong_label(true_label: u8, index: usize) -> u8 {
    let offset = (index % (MNIST_LABEL_CLASS_COUNT - 1)) as u8 + 1;
    (true_label + offset) % MNIST_LABEL_CLASS_COUNT as u8
}

fn conditioned_slot_activation(
    pixels: &[f32; MNIST_IMAGE_PIXELS],
    slot: usize,
    candidate_label: u8,
) -> f32 {
    if slot < LABEL_SLOT_OFFSET {
        return pixels[slot];
    }

    if slot < CONDITIONED_INPUT_NODE_COUNT {
        let label_slot = slot - LABEL_SLOT_OFFSET;
        return if label_slot == usize::from(candidate_label) {
            LABEL_ON_ACTIVATION
        } else {
            LABEL_OFF_ACTIVATION
        };
    }

    0.0
}

#[cfg(test)]
mod tests {
    use hecs::World;

    use super::*;
    use crate::data::mnist_loader::{normalize_mnist_byte, MnistDataset, MNIST_COLS, MNIST_ROWS};

    const EPSILON: f32 = 1.0e-5;

    #[test]
    fn inject_data_system_writes_pixel_and_label_slots_and_toggles_phase() {
        let dataset = MnistDataset::from_raw_bytes(
            build_image_idx(&[sample_with_hot_pixel(0, 255), sample_with_hot_pixel(3, 255)])
                .into_boxed_slice(),
            build_label_idx(&[1, 8]).into_boxed_slice(),
        )
        .expect("construct dataset");

        let mut stream = ContrastiveDataStream::new(&dataset).expect("stream");
        let mut world = World::new();
        for index in 0..(CONDITIONED_INPUT_NODE_COUNT + 2) {
            world.spawn((InputNode, StableNodeIndex::new(index), NodeState::new(-1.0)));
        }
        let mut phase = SimulationPhase::Positive;

        inject_data_system(&mut world, &mut stream, &mut phase).expect("positive inject");
        assert_eq!(phase, SimulationPhase::Negative);
        assert!(
            (world
                .get::<&NodeState>(entity_for_input_slot(&world, 0))
                .expect("pixel slot 0")
                .activation
                - normalize_mnist_byte(255))
            .abs()
                < EPSILON
        );
        assert!(
            (world
                .get::<&NodeState>(entity_for_input_slot(&world, 3))
                .expect("pixel slot 3")
                .activation
                - normalize_mnist_byte(0))
            .abs()
                < EPSILON
        );
        assert_eq!(
            world
                .get::<&NodeState>(entity_for_input_slot(&world, LABEL_SLOT_OFFSET + 1))
                .expect("true label slot")
                .activation,
            LABEL_ON_ACTIVATION
        );
        assert_eq!(
            world
                .get::<&NodeState>(entity_for_input_slot(&world, LABEL_SLOT_OFFSET))
                .expect("off label slot")
                .activation,
            LABEL_OFF_ACTIVATION
        );
        assert_eq!(
            world
                .get::<&NodeState>(entity_for_input_slot(
                    &world,
                    CONDITIONED_INPUT_NODE_COUNT + 1,
                ))
                .expect("extra input slot")
                .activation,
            0.0
        );

        inject_data_system(&mut world, &mut stream, &mut phase).expect("negative inject");
        assert_eq!(phase, SimulationPhase::Positive);
        let wrong_label = deterministic_wrong_label(1, 0);
        assert_eq!(
            world
                .get::<&NodeState>(entity_for_input_slot(&world, LABEL_SLOT_OFFSET + 1))
                .expect("true label off during negative phase")
                .activation,
            LABEL_OFF_ACTIVATION
        );
        assert_eq!(
            world
                .get::<&NodeState>(entity_for_input_slot(
                    &world,
                    LABEL_SLOT_OFFSET + usize::from(wrong_label),
                ))
                .expect("wrong label slot")
                .activation,
            LABEL_ON_ACTIVATION
        );
    }

    #[test]
    fn negative_samples_are_deterministic_and_never_match_the_true_label() {
        let dataset = MnistDataset::from_raw_bytes(
            build_image_idx(&[sample_with_hot_pixel(0, 255), sample_with_hot_pixel(3, 255)])
                .into_boxed_slice(),
            build_label_idx(&[1, 8]).into_boxed_slice(),
        )
        .expect("construct dataset");
        let mut first_pixels = [0.0f32; MNIST_IMAGE_PIXELS];
        let mut second_pixels = [0.0f32; MNIST_IMAGE_PIXELS];

        let first = fill_negative_sample(&dataset, 0, &mut first_pixels).expect("first negative");
        let second =
            fill_negative_sample(&dataset, 0, &mut second_pixels).expect("second negative");

        assert_eq!(first, second);
        assert_eq!(first_pixels, second_pixels);
        assert_ne!(first.true_label, first.candidate_label);
        assert_eq!(first.index, 0);
    }

    #[test]
    fn positive_and_negative_sampling_share_the_real_image_but_not_the_candidate_label() {
        let dataset = MnistDataset::from_raw_bytes(
            build_image_idx(&[sample_with_hot_pixel(0, 255), sample_with_hot_pixel(3, 255)])
                .into_boxed_slice(),
            build_label_idx(&[1, 8]).into_boxed_slice(),
        )
        .expect("construct dataset");
        let mut positive_pixels = [0.0f32; MNIST_IMAGE_PIXELS];
        let mut negative_pixels = [0.0f32; MNIST_IMAGE_PIXELS];

        let positive = fill_positive_sample(&dataset, 0, &mut positive_pixels).expect("positive");
        let negative = fill_negative_sample(&dataset, 0, &mut negative_pixels).expect("negative");

        assert_eq!(positive.index, negative.index);
        assert_eq!(positive.true_label, negative.true_label);
        assert_eq!(positive_pixels, negative_pixels);
        assert_ne!(positive.candidate_label, negative.candidate_label);
    }

    fn entity_for_input_slot(world: &World, slot: usize) -> Entity {
        world
            .query::<&StableNodeIndex>()
            .iter()
            .find_map(|(entity, stable_index)| (stable_index.index == slot).then_some(entity))
            .expect("slot entity should exist")
    }

    fn sample_with_hot_pixel(pixel_index: usize, value: u8) -> [u8; MNIST_IMAGE_PIXELS] {
        let mut pixels = [0u8; MNIST_IMAGE_PIXELS];
        pixels[pixel_index] = value;
        pixels
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
