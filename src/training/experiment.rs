use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ecs_runtime::systems::ActivationKind;
use crate::experiment::ExperimentGoodness;

use super::cli::TrainingConfig;
use super::eval::EvaluationSummary;
use super::world::WorldSummary;

const RUNS_ROOT: &str = "runs";
const MANIFEST_FILE: &str = "manifest.json";
const METRICS_FILE: &str = "metrics.jsonl";
const CHECKPOINTS_DIR: &str = "checkpoints";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ConditioningMode {
    #[default]
    LegacyUnlabeledContrastive,
    LabelConditionedFf,
}

#[derive(Debug, Clone, Copy)]
pub struct EffectiveRunSettings {
    pub graph_node_count: usize,
    pub effective_graph_search_limit: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointSettings {
    pub checkpoint_every: Option<usize>,
    pub final_checkpoint_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointArtifact {
    pub path: PathBuf,
    pub epoch: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunManifest {
    pub run_id: String,
    pub run_dir: PathBuf,
    pub created_at_unix_seconds: u64,
    pub updated_at_unix_seconds: u64,
    pub starting_epoch: usize,
    pub epochs: usize,
    pub train_images: PathBuf,
    pub train_labels: PathBuf,
    pub test_images: Option<PathBuf>,
    pub test_labels: Option<PathBuf>,
    pub graph_node_count: usize,
    pub effective_graph_search_limit: Option<u64>,
    pub activation_kind: ActivationKind,
    pub learning_rate: f32,
    pub weight_seed: u64,
    pub weight_init_scale: f32,
    #[serde(default)]
    pub conditioning_mode: ConditioningMode,
    pub eval_every: usize,
    pub checkpoint: CheckpointSettings,
    pub resume_source: Option<PathBuf>,
    pub latest_checkpoint: Option<CheckpointArtifact>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricsRecord {
    pub epoch: usize,
    pub elapsed_seconds: f64,
    pub mean_abs_activation: f32,
    pub mean_squared_activation: f32,
    pub mean_abs_weight: f32,
    #[serde(default)]
    pub positive_mean_world_goodness: Option<f32>,
    #[serde(default)]
    pub negative_mean_world_goodness: Option<f32>,
    #[serde(default)]
    pub world_goodness_separation: Option<f32>,
    #[serde(default)]
    pub accuracy: Option<f32>,
    #[serde(default)]
    pub mean_correct_goodness: Option<f32>,
    #[serde(default)]
    pub mean_best_wrong_goodness: Option<f32>,
    #[serde(default)]
    pub mean_margin: Option<f32>,
}

impl MetricsRecord {
    pub fn from_summaries(
        epoch: usize,
        elapsed_seconds: f64,
        world: WorldSummary,
        experiment_goodness: &ExperimentGoodness,
        evaluation: Option<&EvaluationSummary>,
    ) -> Self {
        Self {
            epoch,
            elapsed_seconds,
            mean_abs_activation: world.mean_abs_activation,
            mean_squared_activation: world.mean_squared_activation,
            mean_abs_weight: world.mean_abs_weight,
            positive_mean_world_goodness: Some(experiment_goodness.positive_mean_world_goodness),
            negative_mean_world_goodness: Some(experiment_goodness.negative_mean_world_goodness),
            world_goodness_separation: Some(experiment_goodness.goodness_separation),
            accuracy: evaluation.map(|summary| summary.accuracy),
            mean_correct_goodness: evaluation.map(|summary| summary.mean_correct_goodness),
            mean_best_wrong_goodness: evaluation.map(|summary| summary.mean_best_wrong_goodness),
            mean_margin: evaluation.map(|summary| summary.mean_margin),
        }
    }
}

pub struct ExperimentRun {
    root: PathBuf,
    manifest_path: PathBuf,
    checkpoints_dir: PathBuf,
    metrics_sink: MetricsSink,
    manifest: RunManifest,
    starting_epoch: usize,
    elapsed_offset_seconds: f64,
}

impl ExperimentRun {
    pub fn prepare(
        config: &TrainingConfig,
        effective: EffectiveRunSettings,
    ) -> Result<Self, ExperimentError> {
        let now = now_unix_seconds()?;
        let source_manifest = load_source_manifest(config)?;
        let resume_epoch =
            determine_resume_epoch(config.load_checkpoint.as_deref(), source_manifest.as_ref());
        let inferred_checkpoint_run = config
            .load_checkpoint
            .as_deref()
            .and_then(infer_run_dir_from_checkpoint_path);

        let candidate_root = config
            .run_dir
            .clone()
            .or_else(|| inferred_checkpoint_run.clone());

        let (root, append_existing_metrics, existing_manifest, last_metric) = match candidate_root {
            Some(candidate_root) => {
                let manifest_path = candidate_root.join(MANIFEST_FILE);
                let metrics_path = candidate_root.join(METRICS_FILE);
                let existing_manifest = if manifest_path.exists() {
                    Some(load_manifest(&manifest_path)?)
                } else {
                    None
                };
                let last_metric = read_last_metrics_record(&metrics_path)?;
                let has_existing_state = candidate_root.exists()
                    && (manifest_path.exists()
                        || metrics_path.exists()
                        || candidate_root.join(CHECKPOINTS_DIR).exists());

                if config.load_checkpoint.is_none() {
                    if has_existing_state {
                        return Err(ExperimentError::ExistingRunDirectory {
                            path: candidate_root,
                        });
                    }

                    (candidate_root, false, existing_manifest, last_metric)
                } else if let Some(last_metric) = &last_metric {
                    if let Some(resume_epoch) = resume_epoch {
                        if last_metric.epoch > resume_epoch {
                            if config.run_dir.is_some() {
                                return Err(ExperimentError::RunDirectoryAlreadyAhead {
                                    path: candidate_root,
                                    last_metric_epoch: last_metric.epoch,
                                    resume_epoch,
                                });
                            }

                            (allocate_default_run_dir()?, false, None, None)
                        } else {
                            (
                                candidate_root,
                                true,
                                existing_manifest,
                                Some(last_metric.clone()),
                            )
                        }
                    } else if config.run_dir.is_some() {
                        return Err(ExperimentError::ResumeEpochUnknown {
                            checkpoint: config
                                .load_checkpoint
                                .clone()
                                .expect("load checkpoint exists in this branch"),
                        });
                    } else {
                        (allocate_default_run_dir()?, false, None, None)
                    }
                } else {
                    (candidate_root, true, existing_manifest, None)
                }
            }
            None => (allocate_default_run_dir()?, false, None, None),
        };

        fs::create_dir_all(root.join(CHECKPOINTS_DIR)).map_err(|source| ExperimentError::Io {
            path: root.join(CHECKPOINTS_DIR),
            source,
        })?;

        let starting_epoch = resume_epoch.unwrap_or(0);
        let elapsed_offset_seconds = if append_existing_metrics {
            last_metric
                .as_ref()
                .map(|record| record.elapsed_seconds)
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let manifest_path = root.join(MANIFEST_FILE);
        let metrics_path = root.join(METRICS_FILE);
        let run_id = existing_manifest
            .as_ref()
            .map(|manifest| manifest.run_id.clone())
            .unwrap_or_else(|| {
                root.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("run")
                    .to_owned()
            });
        let created_at_unix_seconds = existing_manifest
            .as_ref()
            .map(|manifest| manifest.created_at_unix_seconds)
            .unwrap_or(now);
        let weight_seed = if config.load_checkpoint.is_some() {
            source_manifest
                .as_ref()
                .map(|manifest| manifest.weight_seed)
                .unwrap_or(config.weight_seed)
        } else {
            config.weight_seed
        };
        let weight_init_scale = if config.load_checkpoint.is_some() {
            source_manifest
                .as_ref()
                .map(|manifest| manifest.weight_init_scale)
                .unwrap_or(config.weight_init_scale)
        } else {
            config.weight_init_scale
        };
        let effective_graph_search_limit = effective.effective_graph_search_limit.or_else(|| {
            source_manifest
                .as_ref()
                .and_then(|manifest| manifest.effective_graph_search_limit)
        });

        let manifest = RunManifest {
            run_id,
            run_dir: root.clone(),
            created_at_unix_seconds,
            updated_at_unix_seconds: now,
            starting_epoch,
            epochs: config.epochs,
            train_images: config.train.images.clone(),
            train_labels: config.train.labels.clone(),
            test_images: config.test.as_ref().map(|dataset| dataset.images.clone()),
            test_labels: config.test.as_ref().map(|dataset| dataset.labels.clone()),
            graph_node_count: effective.graph_node_count,
            effective_graph_search_limit,
            activation_kind: config.activation_kind,
            learning_rate: config.learning_rate,
            weight_seed,
            weight_init_scale,
            conditioning_mode: ConditioningMode::LabelConditionedFf,
            eval_every: config.eval_every,
            checkpoint: CheckpointSettings {
                checkpoint_every: config.checkpoint_every,
                final_checkpoint_path: config.save_checkpoint.clone(),
            },
            resume_source: config.load_checkpoint.clone(),
            latest_checkpoint: existing_manifest.and_then(|manifest| manifest.latest_checkpoint),
        };

        write_manifest(&manifest_path, &manifest)?;
        let metrics_sink = MetricsSink::open(metrics_path.clone())?;

        Ok(Self {
            root: root.clone(),
            manifest_path,
            checkpoints_dir: root.join(CHECKPOINTS_DIR),
            metrics_sink,
            manifest,
            starting_epoch,
            elapsed_offset_seconds,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    pub fn metrics_path(&self) -> &Path {
        self.metrics_sink.path()
    }

    pub fn starting_epoch(&self) -> usize {
        self.starting_epoch
    }

    pub fn elapsed_offset_seconds(&self) -> f64 {
        self.elapsed_offset_seconds
    }

    pub fn should_save_periodic_checkpoint(&self, epoch: usize, final_epoch: usize) -> bool {
        self.manifest
            .checkpoint
            .checkpoint_every
            .map(|every| epoch.is_multiple_of(every) || epoch == final_epoch)
            .unwrap_or(false)
    }

    pub fn checkpoint_path_for_epoch(&self, epoch: usize) -> PathBuf {
        self.checkpoints_dir.join(format!("epoch-{epoch:06}.json"))
    }

    pub fn append_metrics(&mut self, record: &MetricsRecord) -> Result<(), ExperimentError> {
        self.metrics_sink.append(record)
    }

    pub fn record_checkpoint<P: AsRef<Path>>(
        &mut self,
        path: P,
        epoch: usize,
    ) -> Result<(), ExperimentError> {
        self.manifest.latest_checkpoint = Some(CheckpointArtifact {
            path: path.as_ref().to_path_buf(),
            epoch,
        });
        self.manifest.updated_at_unix_seconds = now_unix_seconds()?;
        write_manifest(&self.manifest_path, &self.manifest)
    }

    pub fn latest_checkpoint_path(&self) -> Option<&Path> {
        self.manifest
            .latest_checkpoint
            .as_ref()
            .map(|artifact| artifact.path.as_path())
    }
}

pub struct MetricsSink {
    path: PathBuf,
    file: File,
}

impl MetricsSink {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, ExperimentError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| ExperimentError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| ExperimentError::Io {
                path: path.clone(),
                source,
            })?;

        Ok(Self { path, file })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn append(&mut self, record: &MetricsRecord) -> Result<(), ExperimentError> {
        let json = serde_json::to_string(record).map_err(|source| ExperimentError::Serialize {
            path: self.path.clone(),
            source,
        })?;
        writeln!(self.file, "{json}").map_err(|source| ExperimentError::Io {
            path: self.path.clone(),
            source,
        })?;
        self.file.flush().map_err(|source| ExperimentError::Io {
            path: self.path.clone(),
            source,
        })
    }
}

#[derive(Debug, Error)]
pub enum ExperimentError {
    #[error("failed to read or write experiment file {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to serialize experiment file {path}: {source}")]
    Serialize {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to deserialize experiment file {path}: {source}")]
    Deserialize {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("run directory {path} already contains manifest, metrics, or checkpoints; pass --load-checkpoint to resume or choose a new --run-dir")]
    ExistingRunDirectory { path: PathBuf },
    #[error("run directory {path} already has metrics through epoch {last_metric_epoch}, which is newer than the requested resume checkpoint epoch {resume_epoch}")]
    RunDirectoryAlreadyAhead {
        path: PathBuf,
        last_metric_epoch: usize,
        resume_epoch: usize,
    },
    #[error("could not determine a resume epoch for checkpoint {checkpoint}; use a run-directory checkpoint named epoch-<N>.json or resume into a fresh run directory")]
    ResumeEpochUnknown { checkpoint: PathBuf },
    #[error("system clock is earlier than the Unix epoch")]
    InvalidSystemTime,
    #[error("could not allocate a unique default run directory under {root}")]
    CouldNotAllocateDefaultRunDirectory { root: PathBuf },
}

pub fn format_default_run_dir(unix_seconds: u64, short_id: &str) -> PathBuf {
    PathBuf::from(RUNS_ROOT).join(format!("{unix_seconds}-{short_id}"))
}

fn allocate_default_run_dir() -> Result<PathBuf, ExperimentError> {
    let root = PathBuf::from(RUNS_ROOT);
    for _ in 0..32 {
        let candidate = format_default_run_dir(now_unix_seconds()?, &random_short_id());
        if !candidate.exists() {
            return Ok(candidate);
        }
    }

    Err(ExperimentError::CouldNotAllocateDefaultRunDirectory { root })
}

fn now_unix_seconds() -> Result<u64, ExperimentError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|_| ExperimentError::InvalidSystemTime)
}

fn random_short_id() -> String {
    format!("{:06x}", rand::thread_rng().gen::<u32>() & 0x00ff_ffff)
}

fn load_source_manifest(config: &TrainingConfig) -> Result<Option<RunManifest>, ExperimentError> {
    if let Some(run_dir) = &config.run_dir {
        let manifest_path = run_dir.join(MANIFEST_FILE);
        if manifest_path.exists() {
            return load_manifest(&manifest_path).map(Some);
        }
    }

    if let Some(checkpoint) = config
        .load_checkpoint
        .as_deref()
        .and_then(infer_run_dir_from_checkpoint_path)
    {
        let manifest_path = checkpoint.join(MANIFEST_FILE);
        if manifest_path.exists() {
            return load_manifest(&manifest_path).map(Some);
        }
    }

    Ok(None)
}

fn write_manifest(path: &Path, manifest: &RunManifest) -> Result<(), ExperimentError> {
    let bytes =
        serde_json::to_vec_pretty(manifest).map_err(|source| ExperimentError::Serialize {
            path: path.to_path_buf(),
            source,
        })?;
    fs::write(path, bytes).map_err(|source| ExperimentError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn load_manifest(path: &Path) -> Result<RunManifest, ExperimentError> {
    let bytes = fs::read(path).map_err(|source| ExperimentError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| ExperimentError::Deserialize {
        path: path.to_path_buf(),
        source,
    })
}

fn read_last_metrics_record(path: &Path) -> Result<Option<MetricsRecord>, ExperimentError> {
    if !path.exists() {
        return Ok(None);
    }

    let file = File::open(path).map_err(|source| ExperimentError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let reader = BufReader::new(file);
    let mut last_line = None;

    for line in reader.lines() {
        let line = line.map_err(|source| ExperimentError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if !line.trim().is_empty() {
            last_line = Some(line);
        }
    }

    match last_line {
        Some(line) => {
            serde_json::from_str(&line)
                .map(Some)
                .map_err(|source| ExperimentError::Deserialize {
                    path: path.to_path_buf(),
                    source,
                })
        }
        None => Ok(None),
    }
}

fn determine_resume_epoch(
    checkpoint: Option<&Path>,
    manifest: Option<&RunManifest>,
) -> Option<usize> {
    let checkpoint = checkpoint?;

    parse_checkpoint_epoch(checkpoint).or_else(|| {
        manifest.and_then(|manifest| {
            manifest
                .latest_checkpoint
                .as_ref()
                .filter(|artifact| artifact.path == checkpoint)
                .map(|artifact| artifact.epoch)
        })
    })
}

fn parse_checkpoint_epoch(path: &Path) -> Option<usize> {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| stem.strip_prefix("epoch-"))
        .and_then(|epoch| epoch.parse::<usize>().ok())
}

fn infer_run_dir_from_checkpoint_path(path: &Path) -> Option<PathBuf> {
    let checkpoint_dir = path.parent()?;
    if checkpoint_dir.file_name()?.to_str()? != CHECKPOINTS_DIR {
        return None;
    }

    checkpoint_dir.parent().map(Path::to_path_buf)
}

#[cfg(test)]
mod tests {
    use std::env;

    use super::*;
    use crate::ecs_runtime::systems::ActivationKind;
    use super::super::cli::DatasetPaths;

    #[test]
    fn default_run_dir_format_is_stable() {
        assert_eq!(
            format_default_run_dir(1_710_000_000, "abc123"),
            PathBuf::from("runs/1710000000-abc123")
        );
    }

    #[test]
    fn metrics_sink_writes_jsonl_records() {
        let dir = unique_test_dir("metrics");
        fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("metrics.jsonl");
        let mut sink = MetricsSink::open(&path).expect("open metrics sink");
        let record = MetricsRecord {
            epoch: 3,
            elapsed_seconds: 12.5,
            mean_abs_activation: 0.25,
            mean_squared_activation: 0.5,
            mean_abs_weight: 0.75,
            positive_mean_world_goodness: Some(1.6),
            negative_mean_world_goodness: Some(1.1),
            world_goodness_separation: Some(0.5),
            accuracy: Some(0.75),
            mean_correct_goodness: Some(1.2),
            mean_best_wrong_goodness: Some(0.8),
            mean_margin: Some(0.4),
        };

        sink.append(&record).expect("append metrics");

        let line = fs::read_to_string(&path)
            .expect("read metrics")
            .lines()
            .next()
            .expect("metrics line")
            .to_owned();
        let parsed: MetricsRecord = serde_json::from_str(&line).expect("parse metrics");

        assert_eq!(parsed, record);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn infers_run_dir_from_checkpoint_layout() {
        let path = PathBuf::from("runs/1710000000-abc123/checkpoints/epoch-000005.json");
        assert_eq!(
            infer_run_dir_from_checkpoint_path(&path),
            Some(PathBuf::from("runs/1710000000-abc123"))
        );
        assert_eq!(parse_checkpoint_epoch(&path), Some(5));
    }

    #[test]
    fn prepared_run_directory_writes_conditioning_mode_and_accuracy_metrics() {
        let dir = unique_test_dir("run-dir");
        let config = TrainingConfig {
            train: DatasetPaths {
                images: PathBuf::from("train-images"),
                labels: PathBuf::from("train-labels"),
            },
            test: None,
            epochs: 1,
            learning_rate: 1.0e-3,
            activation_kind: ActivationKind::Relu,
            graph_node_count: 794,
            graph_search_limit: Some(123),
            weight_seed: 7,
            weight_init_scale: 0.05,
            run_dir: Some(dir.clone()),
            checkpoint_every: None,
            eval_every: 1,
            load_checkpoint: None,
            save_checkpoint: None,
        };

        let mut run = ExperimentRun::prepare(
            &config,
            EffectiveRunSettings {
                graph_node_count: 794,
                effective_graph_search_limit: Some(123),
            },
        )
        .expect("prepare experiment");
        let record = MetricsRecord {
            epoch: 1,
            elapsed_seconds: 0.5,
            mean_abs_activation: 0.25,
            mean_squared_activation: 0.5,
            mean_abs_weight: 0.75,
            positive_mean_world_goodness: Some(1.6),
            negative_mean_world_goodness: Some(1.1),
            world_goodness_separation: Some(0.5),
            accuracy: Some(0.75),
            mean_correct_goodness: Some(1.2),
            mean_best_wrong_goodness: Some(0.8),
            mean_margin: Some(0.4),
        };
        run.append_metrics(&record).expect("append metrics");

        let manifest: serde_json::Value = serde_json::from_slice(
            &fs::read(run.manifest_path()).expect("read manifest"),
        )
        .expect("parse manifest");
        assert_eq!(manifest["conditioning_mode"], "label_conditioned_ff");

        let metrics_line = fs::read_to_string(run.metrics_path())
            .expect("read metrics")
            .lines()
            .next()
            .expect("metrics line")
            .to_owned();
        let metrics: serde_json::Value =
            serde_json::from_str(&metrics_line).expect("parse metrics line");
        assert_eq!(metrics["accuracy"], 0.75);
        assert_eq!(metrics["mean_correct_goodness"], 1.2);
        assert_eq!(metrics["mean_best_wrong_goodness"], 0.8);
        assert_eq!(metrics["mean_margin"], 0.4);

        let _ = fs::remove_dir_all(dir);
    }

    fn unique_test_dir(stem: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should advance")
            .as_nanos();
        env::temp_dir().join(format!("golem-engine-{stem}-{unique}"))
    }
}
