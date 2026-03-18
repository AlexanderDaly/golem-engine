//! Native IDX parsing for the MNIST contrastive stream.
//!
//! The loader performs all file I/O up front and retains the raw file contents in two
//! contiguous allocations. Hot-path training code can then borrow image slices directly
//! and normalize into caller-provided buffers without any further file-system traffic.

use std::fs;
use std::io;
use std::iter::FusedIterator;
use std::path::{Path, PathBuf};

use thiserror::Error;

pub const MNIST_ROWS: usize = 28;
pub const MNIST_COLS: usize = 28;
pub const MNIST_IMAGE_PIXELS: usize = MNIST_ROWS * MNIST_COLS;
pub const MNIST_IMAGE_MAGIC: u32 = 2_051;
pub const MNIST_LABEL_MAGIC: u32 = 2_049;
pub const MNIST_NORMALIZATION_MEAN: f32 = 0.1307;
pub const MNIST_NORMALIZATION_STDDEV: f32 = 0.3081;

const IMAGE_HEADER_BYTES: usize = 16;
const LABEL_HEADER_BYTES: usize = 8;
const UNIT_SCALE: f32 = 1.0 / 255.0;

#[derive(Debug)]
pub struct MnistDataset {
    image_bytes: Box<[u8]>,
    label_bytes: Box<[u8]>,
    image_offset: usize,
    label_offset: usize,
    image_count: usize,
}

impl MnistDataset {
    pub fn load<PI, PL>(image_path: PI, label_path: PL) -> Result<Self, MnistLoadError>
    where
        PI: AsRef<Path>,
        PL: AsRef<Path>,
    {
        let image_path = image_path.as_ref().to_path_buf();
        let label_path = label_path.as_ref().to_path_buf();

        let image_bytes = fs::read(&image_path).map_err(|source| MnistLoadError::Io {
            path: image_path.clone(),
            source,
        })?;
        let label_bytes = fs::read(&label_path).map_err(|source| MnistLoadError::Io {
            path: label_path.clone(),
            source,
        })?;

        Self::from_raw_bytes(
            image_bytes.into_boxed_slice(),
            label_bytes.into_boxed_slice(),
        )
    }

    pub fn from_raw_bytes(
        image_bytes: Box<[u8]>,
        label_bytes: Box<[u8]>,
    ) -> Result<Self, MnistLoadError> {
        if image_bytes.len() < IMAGE_HEADER_BYTES {
            return Err(MnistLoadError::TruncatedFile {
                kind: IdxKind::Images,
                expected_at_least: IMAGE_HEADER_BYTES,
                actual: image_bytes.len(),
            });
        }

        if label_bytes.len() < LABEL_HEADER_BYTES {
            return Err(MnistLoadError::TruncatedFile {
                kind: IdxKind::Labels,
                expected_at_least: LABEL_HEADER_BYTES,
                actual: label_bytes.len(),
            });
        }

        let image_magic = read_u32_be(&image_bytes, 0, IdxKind::Images)?;
        if image_magic != MNIST_IMAGE_MAGIC {
            return Err(MnistLoadError::InvalidMagic {
                kind: IdxKind::Images,
                expected: MNIST_IMAGE_MAGIC,
                found: image_magic,
            });
        }

        let label_magic = read_u32_be(&label_bytes, 0, IdxKind::Labels)?;
        if label_magic != MNIST_LABEL_MAGIC {
            return Err(MnistLoadError::InvalidMagic {
                kind: IdxKind::Labels,
                expected: MNIST_LABEL_MAGIC,
                found: label_magic,
            });
        }

        let image_count = read_u32_be(&image_bytes, 4, IdxKind::Images)? as usize;
        let rows = read_u32_be(&image_bytes, 8, IdxKind::Images)? as usize;
        let cols = read_u32_be(&image_bytes, 12, IdxKind::Images)? as usize;

        if rows != MNIST_ROWS || cols != MNIST_COLS {
            return Err(MnistLoadError::UnsupportedDimensions { rows, cols });
        }

        let declared_image_bytes = image_count
            .checked_mul(MNIST_IMAGE_PIXELS)
            .ok_or(MnistLoadError::ImageCountOverflow { image_count })?;
        let actual_image_bytes = image_bytes.len() - IMAGE_HEADER_BYTES;
        if actual_image_bytes != declared_image_bytes {
            return Err(MnistLoadError::PayloadLengthMismatch {
                kind: IdxKind::Images,
                expected: declared_image_bytes,
                actual: actual_image_bytes,
            });
        }

        let label_count = read_u32_be(&label_bytes, 4, IdxKind::Labels)? as usize;
        let actual_label_bytes = label_bytes.len() - LABEL_HEADER_BYTES;
        if actual_label_bytes != label_count {
            return Err(MnistLoadError::PayloadLengthMismatch {
                kind: IdxKind::Labels,
                expected: label_count,
                actual: actual_label_bytes,
            });
        }

        if image_count != label_count {
            return Err(MnistLoadError::CountMismatch {
                image_count,
                label_count,
            });
        }

        Ok(Self {
            image_bytes,
            label_bytes,
            image_offset: IMAGE_HEADER_BYTES,
            label_offset: LABEL_HEADER_BYTES,
            image_count,
        })
    }

    pub fn len(&self) -> usize {
        self.image_count
    }

    pub fn is_empty(&self) -> bool {
        self.image_count == 0
    }

    pub fn label(&self, index: usize) -> Result<u8, MnistSampleError> {
        self.ensure_in_bounds(index)?;
        Ok(self.label_bytes[self.label_offset + index])
    }

    pub fn image(&self, index: usize) -> Result<MnistImageRef<'_>, MnistSampleError> {
        self.ensure_in_bounds(index)?;

        let start = self.image_offset + index * MNIST_IMAGE_PIXELS;
        let end = start + MNIST_IMAGE_PIXELS;
        let pixels = self.image_bytes[start..end]
            .try_into()
            .expect("validated 28x28 payload slices always fit the fixed array");

        Ok(MnistImageRef {
            pixels,
            label: self.label_bytes[self.label_offset + index],
            index,
        })
    }

    pub fn fill_normalized_image(
        &self,
        index: usize,
        output: &mut [f32; MNIST_IMAGE_PIXELS],
    ) -> Result<u8, MnistSampleError> {
        let sample = self.image(index)?;

        for (destination, pixel) in output.iter_mut().zip(sample.pixels.iter().copied()) {
            *destination = normalize_mnist_byte(pixel);
        }

        Ok(sample.label)
    }

    pub fn normalized_image(
        &self,
        index: usize,
    ) -> Result<NormalizedMnistSample, MnistSampleError> {
        let mut pixels = [0.0f32; MNIST_IMAGE_PIXELS];
        let label = self.fill_normalized_image(index, &mut pixels)?;

        Ok(NormalizedMnistSample {
            pixels,
            label,
            index,
        })
    }

    pub fn normalized_samples(&self) -> NormalizedMnistIter<'_> {
        NormalizedMnistIter {
            dataset: self,
            next_index: 0,
        }
    }

    fn ensure_in_bounds(&self, index: usize) -> Result<(), MnistSampleError> {
        if index >= self.image_count {
            return Err(MnistSampleError::IndexOutOfBounds {
                index,
                len: self.image_count,
            });
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MnistImageRef<'a> {
    pub pixels: &'a [u8; MNIST_IMAGE_PIXELS],
    pub label: u8,
    pub index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedMnistSample {
    pub pixels: [f32; MNIST_IMAGE_PIXELS],
    pub label: u8,
    pub index: usize,
}

pub struct NormalizedMnistIter<'a> {
    dataset: &'a MnistDataset,
    next_index: usize,
}

impl<'a> Iterator for NormalizedMnistIter<'a> {
    type Item = NormalizedMnistSample;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.dataset.len() {
            return None;
        }

        let sample = self
            .dataset
            .normalized_image(self.next_index)
            .expect("iterator only advances over in-bounds samples");
        self.next_index += 1;
        Some(sample)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.dataset.len().saturating_sub(self.next_index);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for NormalizedMnistIter<'_> {}
impl FusedIterator for NormalizedMnistIter<'_> {}

pub fn normalize_mnist_byte(pixel: u8) -> f32 {
    standardize_unit_pixel(f32::from(pixel) * UNIT_SCALE)
}

pub fn standardize_unit_pixel(pixel: f32) -> f32 {
    (pixel - MNIST_NORMALIZATION_MEAN) / MNIST_NORMALIZATION_STDDEV
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdxKind {
    Images,
    Labels,
}

#[derive(Debug, Error)]
pub enum MnistLoadError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("{kind:?} IDX file is truncated: expected at least {expected_at_least} bytes, found {actual}")]
    TruncatedFile {
        kind: IdxKind,
        expected_at_least: usize,
        actual: usize,
    },
    #[error("failed to parse {kind:?} IDX header")]
    InvalidHeader { kind: IdxKind },
    #[error("{kind:?} IDX magic mismatch: expected {expected}, found {found}")]
    InvalidMagic {
        kind: IdxKind,
        expected: u32,
        found: u32,
    },
    #[error("expected 28x28 MNIST images, found {rows}x{cols}")]
    UnsupportedDimensions { rows: usize, cols: usize },
    #[error("image count {image_count} overflows the declared MNIST payload size")]
    ImageCountOverflow { image_count: usize },
    #[error("{kind:?} payload length mismatch: expected {expected} bytes, found {actual}")]
    PayloadLengthMismatch {
        kind: IdxKind,
        expected: usize,
        actual: usize,
    },
    #[error("image count mismatch: {image_count} images but {label_count} labels")]
    CountMismatch {
        image_count: usize,
        label_count: usize,
    },
}

#[derive(Debug, Error)]
pub enum MnistSampleError {
    #[error("sample index {index} is out of bounds for dataset length {len}")]
    IndexOutOfBounds { index: usize, len: usize },
}

fn read_u32_be(bytes: &[u8], offset: usize, kind: IdxKind) -> Result<u32, MnistLoadError> {
    let header_slice = bytes
        .get(offset..offset + 4)
        .ok_or(MnistLoadError::InvalidHeader { kind })?;

    Ok(u32::from_be_bytes(
        header_slice
            .try_into()
            .expect("exactly four bytes are always present here"),
    ))
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn loads_native_idx_bytes_and_normalizes_without_extra_copies() {
        let image_path = unique_test_path("mnist-images");
        let label_path = unique_test_path("mnist-labels");

        fs::write(&image_path, build_image_idx(&[0u8, 255u8])).expect("write image fixture");
        fs::write(&label_path, build_label_idx(&[3u8, 8u8])).expect("write label fixture");

        let dataset = MnistDataset::load(&image_path, &label_path).expect("load fixture dataset");

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.label(0).expect("label 0"), 3);
        assert_eq!(dataset.label(1).expect("label 1"), 8);

        let first = dataset.normalized_image(0).expect("first sample");
        assert_eq!(first.label, 3);
        assert!(first
            .pixels
            .iter()
            .all(|pixel| (*pixel - normalize_mnist_byte(0)).abs() < 1.0e-6));

        let second = dataset.normalized_samples().nth(1).expect("second sample");
        assert_eq!(second.index, 1);
        assert!(second
            .pixels
            .iter()
            .all(|pixel| (*pixel - normalize_mnist_byte(255)).abs() < 1.0e-6));

        let _ = fs::remove_file(image_path);
        let _ = fs::remove_file(label_path);
    }

    fn build_image_idx(fill_values: &[u8]) -> Vec<u8> {
        let mut bytes =
            Vec::with_capacity(IMAGE_HEADER_BYTES + fill_values.len() * MNIST_IMAGE_PIXELS);
        bytes.extend_from_slice(&MNIST_IMAGE_MAGIC.to_be_bytes());
        bytes.extend_from_slice(&(fill_values.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&(MNIST_ROWS as u32).to_be_bytes());
        bytes.extend_from_slice(&(MNIST_COLS as u32).to_be_bytes());

        for fill in fill_values {
            bytes.extend(std::iter::repeat_n(*fill, MNIST_IMAGE_PIXELS));
        }

        bytes
    }

    fn build_label_idx(labels: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(LABEL_HEADER_BYTES + labels.len());
        bytes.extend_from_slice(&MNIST_LABEL_MAGIC.to_be_bytes());
        bytes.extend_from_slice(&(labels.len() as u32).to_be_bytes());
        bytes.extend_from_slice(labels);
        bytes
    }

    fn unique_test_path(stem: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should advance")
            .as_nanos();
        env::temp_dir().join(format!("golem-engine-{stem}-{unique}.idx"))
    }
}
