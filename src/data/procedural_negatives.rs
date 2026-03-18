//! Procedural negatives for the Forward-Forward contrastive loop.
//!
//! The current negative generator preserves local stroke statistics by combining two
//! real digits with a spatial mask instead of injecting unstructured noise. That forces
//! graph nodes to separate plausible but semantically invalid images from true examples.

use thiserror::Error;

use crate::data::mnist_loader::{
    standardize_unit_pixel, MnistDataset, MnistImageRef, MnistSampleError, MNIST_COLS,
    MNIST_IMAGE_PIXELS, MNIST_ROWS,
};

const SEAM_ROW_TOP: usize = (MNIST_ROWS / 2) - 1;
const SEAM_ROW_BOTTOM: usize = MNIST_ROWS / 2;
const TOP_SEAM_BOTTOM_WEIGHT: f32 = 0.25;
const BOTTOM_SEAM_BOTTOM_WEIGHT: f32 = 0.75;
const PIXEL_SCALE: f32 = 1.0 / 255.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridNegativeSample {
    pub pixels: [f32; MNIST_IMAGE_PIXELS],
    pub upper_index: usize,
    pub lower_index: usize,
    pub upper_label: u8,
    pub lower_label: u8,
}

pub fn generate_hybrid_negative(
    upper: MnistImageRef<'_>,
    lower: MnistImageRef<'_>,
) -> Result<HybridNegativeSample, HybridNegativeError> {
    if upper.index == lower.index {
        return Err(HybridNegativeError::DuplicateSource { index: upper.index });
    }

    if upper.label == lower.label {
        return Err(HybridNegativeError::MatchingLabels {
            upper_index: upper.index,
            lower_index: lower.index,
            label: upper.label,
        });
    }

    let mut pixels = [0.0f32; MNIST_IMAGE_PIXELS];

    for row in 0..MNIST_ROWS {
        let lower_weight = seam_lower_weight(row);
        let upper_weight = 1.0 - lower_weight;
        let row_offset = row * MNIST_COLS;

        for col in 0..MNIST_COLS {
            let index = row_offset + col;
            let upper_pixel = f32::from(upper.pixels[index]);
            let lower_pixel = f32::from(lower.pixels[index]);
            let blended_pixel =
                (upper_pixel * upper_weight + lower_pixel * lower_weight) * PIXEL_SCALE;
            pixels[index] = standardize_unit_pixel(blended_pixel);
        }
    }

    Ok(HybridNegativeSample {
        pixels,
        upper_index: upper.index,
        lower_index: lower.index,
        upper_label: upper.label,
        lower_label: lower.label,
    })
}

pub fn generate_hybrid_negative_from_indices(
    dataset: &MnistDataset,
    upper_index: usize,
    lower_index: usize,
) -> Result<HybridNegativeSample, HybridNegativeError> {
    let upper = dataset.image(upper_index)?;
    let lower = dataset.image(lower_index)?;
    generate_hybrid_negative(upper, lower)
}

#[derive(Debug, Error)]
pub enum HybridNegativeError {
    #[error("sample index lookup failed: {0}")]
    InvalidSample(#[from] MnistSampleError),
    #[error("hybrid negatives require two distinct source images; received index {index} twice")]
    DuplicateSource { index: usize },
    #[error("hybrid negatives require distinct labels; indices {upper_index} and {lower_index} both carry label {label}")]
    MatchingLabels {
        upper_index: usize,
        lower_index: usize,
        label: u8,
    },
}

fn seam_lower_weight(row: usize) -> f32 {
    match row {
        0..SEAM_ROW_TOP => 0.0,
        SEAM_ROW_TOP => TOP_SEAM_BOTTOM_WEIGHT,
        SEAM_ROW_BOTTOM => BOTTOM_SEAM_BOTTOM_WEIGHT,
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::mnist_loader::normalize_mnist_byte;

    #[test]
    fn hybrid_negative_blends_top_and_bottom_halves_with_a_soft_seam() {
        let top = [255u8; MNIST_IMAGE_PIXELS];
        let bottom = [0u8; MNIST_IMAGE_PIXELS];

        let sample = generate_hybrid_negative(
            MnistImageRef {
                pixels: &top,
                label: 3,
                index: 0,
            },
            MnistImageRef {
                pixels: &bottom,
                label: 8,
                index: 1,
            },
        )
        .expect("valid hybrid negative");

        let top_row_start = 0;
        let seam_top_row_start = SEAM_ROW_TOP * MNIST_COLS;
        let seam_bottom_row_start = SEAM_ROW_BOTTOM * MNIST_COLS;
        let bottom_row_start = (MNIST_ROWS - 1) * MNIST_COLS;

        assert_eq!(sample.pixels[top_row_start], normalize_mnist_byte(255));
        assert_eq!(sample.pixels[bottom_row_start], normalize_mnist_byte(0),);
        assert!((sample.pixels[seam_top_row_start] - standardize_unit_pixel(0.75)).abs() < 1.0e-6);
        assert!(
            (sample.pixels[seam_bottom_row_start] - standardize_unit_pixel(0.25)).abs() < 1.0e-6
        );
    }
}
