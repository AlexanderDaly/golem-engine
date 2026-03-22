use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::ops::Range;
use std::thread;
use std::time::Instant;

use hecs::World;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::data::mnist_loader::{MnistDataset, MnistLoadError, MNIST_IMAGE_PIXELS};
use crate::ecs_runtime::checkpoint::GraphNodeCheckpoint;
use crate::ecs_runtime::checkpoint::{save_checkpoint_json, CheckpointError, GraphCheckpoint};
use crate::ecs_runtime::ingestion_system::{
    fill_negative_sample, fill_positive_sample, inject_conditioned_sample, IngestionError,
    SimulationPhase,
};
use crate::ecs_runtime::systems::{
    update_local_weights_forward_forward, update_nodes_forward_forward, ActivationKind,
    NodeUpdateError,
};
use crate::experiment::{ExperimentGoodness, ExperimentGoodnessAccumulator, ExperimentMetricError};

use super::cli::{DatasetPaths, WorkerServerConfig};
use super::eval::{evaluate_forward_forward_range, EvaluationAccumulator};
use super::{
    summarize_world, EffectiveRunSettings, EvaluationError, ExperimentRun, MetricsRecord,
    PreparedTrainingState, TrainingConfig, TrainingRunError, TrainingRunOutput,
};

const DISTRIBUTED_PROTOCOL_VERSION: u32 = 1;
const MAX_MESSAGE_BYTES: usize = 512 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum DistributedRuntimeError {
    #[error("no distributed workers were configured")]
    NoWorkersConfigured,
    #[error("failed to bind distributed worker listener on {addr}: {source}")]
    Bind {
        addr: String,
        #[source]
        source: io::Error,
    },
    #[error("failed to accept a distributed worker connection on {addr}: {source}")]
    Accept {
        addr: String,
        #[source]
        source: io::Error,
    },
    #[error("distributed transport failed for {context}: {source}")]
    Transport {
        context: String,
        #[source]
        source: io::Error,
    },
    #[error("distributed protocol serialization failed: {0}")]
    Protocol(#[from] serde_json::Error),
    #[error("distributed protocol version mismatch: local={local} remote={remote}")]
    ProtocolVersionMismatch { local: u32, remote: u32 },
    #[error("worker {worker} returned an unexpected response")]
    UnexpectedWorkerResponse { worker: String },
    #[error("worker {worker} reported an error: {message}")]
    WorkerError { worker: String, message: String },
    #[error("distributed worker thread panicked")]
    WorkerThreadPanicked,
    #[error("distributed shard {start}..{end} is invalid for dataset length {dataset_len}")]
    InvalidShard {
        start: usize,
        end: usize,
        dataset_len: usize,
    },
    #[error("distributed averaging requires at least one worker result")]
    NoWorkerResults,
    #[error("worker checkpoints are incompatible at node {node}: {reason}")]
    IncompatibleCheckpoint { node: usize, reason: &'static str },
    #[error("distributed protocol message exceeds the {max_bytes}-byte safety limit")]
    MessageTooLarge { max_bytes: usize },
    #[error("failed to load MNIST dataset on a distributed worker: {0}")]
    MnistLoad(#[from] MnistLoadError),
    #[error("checkpoint operation failed during distributed execution: {0}")]
    Checkpoint(#[from] CheckpointError),
    #[error("data ingestion failed during distributed execution: {0}")]
    Ingestion(#[from] IngestionError),
    #[error("forward-forward update failed during distributed execution: {0}")]
    NodeUpdate(#[from] NodeUpdateError),
    #[error("failed to compute experiment goodness during distributed execution: {0}")]
    ExperimentMetric(#[from] ExperimentMetricError),
    #[error("evaluation failed during distributed execution: {0}")]
    Evaluation(#[from] EvaluationError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerRequest {
    protocol_version: u32,
    payload: WorkerRequestPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum WorkerRequestPayload {
    Ping,
    TrainShard(WorkerTrainShardRequest),
    EvaluateShard(WorkerEvaluateShardRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerResponse {
    protocol_version: u32,
    payload: WorkerResponsePayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum WorkerResponsePayload {
    Pong,
    TrainShard(WorkerTrainShardResponse),
    EvaluateShard(WorkerEvaluateShardResponse),
    Error(WorkerErrorResponse),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerErrorResponse {
    message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct ShardRange {
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct ShardExperimentMetrics {
    positive_mean_world_goodness: f32,
    negative_mean_world_goodness: f32,
    positive_samples: usize,
    negative_samples: usize,
}

impl From<ExperimentGoodness> for ShardExperimentMetrics {
    fn from(value: ExperimentGoodness) -> Self {
        Self {
            positive_mean_world_goodness: value.positive_mean_world_goodness,
            negative_mean_world_goodness: value.negative_mean_world_goodness,
            positive_samples: value.positive_samples,
            negative_samples: value.negative_samples,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerTrainShardRequest {
    checkpoint: GraphCheckpoint,
    dataset: DatasetPaths,
    activation_kind: ActivationKind,
    learning_rate: f32,
    shard: ShardRange,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerTrainShardResponse {
    checkpoint: GraphCheckpoint,
    experiment_goodness: ShardExperimentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerEvaluateShardRequest {
    checkpoint: GraphCheckpoint,
    dataset: DatasetPaths,
    activation_kind: ActivationKind,
    shard: ShardRange,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerEvaluateShardResponse {
    accumulator: EvaluationAccumulator,
}

#[derive(Default)]
struct WorkerDatasetCache {
    datasets: HashMap<DatasetPaths, MnistDataset>,
}

impl WorkerDatasetCache {
    fn load(&mut self, paths: &DatasetPaths) -> Result<&MnistDataset, DistributedRuntimeError> {
        if !self.datasets.contains_key(paths) {
            let dataset = MnistDataset::load(&paths.images, &paths.labels)?;
            self.datasets.insert(paths.clone(), dataset);
        }

        Ok(self
            .datasets
            .get(paths)
            .expect("dataset cache entry should exist after load"))
    }
}

pub fn run_worker_server(config: WorkerServerConfig) -> Result<(), DistributedRuntimeError> {
    let listener =
        TcpListener::bind(&config.listen_addr).map_err(|source| DistributedRuntimeError::Bind {
            addr: config.listen_addr.clone(),
            source,
        })?;
    let mut dataset_cache = WorkerDatasetCache::default();

    println!("distributed worker listening on {}", config.listen_addr);

    loop {
        let (stream, peer_addr) =
            listener
                .accept()
                .map_err(|source| DistributedRuntimeError::Accept {
                    addr: config.listen_addr.clone(),
                    source,
                })?;
        if let Err(error) =
            handle_worker_connection(stream, &peer_addr.to_string(), &mut dataset_cache)
        {
            eprintln!("worker request failed: {error}");
        }
    }
}

pub(crate) fn run_distributed_training(
    config: TrainingConfig,
    prepared: PreparedTrainingState,
) -> Result<TrainingRunOutput, TrainingRunError> {
    if config.distributed_workers.is_empty() {
        return Err(DistributedRuntimeError::NoWorkersConfigured.into());
    }

    let PreparedTrainingState {
        train_dataset,
        test_dataset,
        mut world,
        mut phase,
        effective_graph_search_limit,
        world_node_count,
        input_node_count,
    } = prepared;

    ping_workers(&config.distributed_workers)?;

    let mut experiment = ExperimentRun::prepare(
        &config,
        EffectiveRunSettings {
            graph_node_count: world_node_count,
            effective_graph_search_limit,
        },
    )?;
    let final_epoch = experiment.starting_epoch() + config.epochs;
    let eval_dataset = test_dataset.as_ref().unwrap_or(&train_dataset);
    let eval_paths = config.test.as_ref().unwrap_or(&config.train);
    let ticks_per_epoch =
        train_dataset
            .len()
            .checked_mul(2)
            .ok_or(TrainingRunError::TickCountOverflow {
                dataset_len: train_dataset.len(),
                epochs: config.epochs,
            })?;
    let total_ticks =
        ticks_per_epoch
            .checked_mul(config.epochs)
            .ok_or(TrainingRunError::TickCountOverflow {
                dataset_len: train_dataset.len(),
                epochs: config.epochs,
            })?;
    let run_started = Instant::now();

    println!("run directory: {}", experiment.root().display());
    println!(
        "starting distributed training: start_epoch={} end_epoch={} epochs_this_invocation={} ticks_per_epoch={} total_ticks={} activation={} learning_rate={} graph_node_count={} input_node_count={} weight_seed={} weight_init_scale={} eval_every={} checkpoint_every={} workers={} strategy=federated_averaging",
        experiment.starting_epoch(),
        final_epoch,
        config.epochs,
        ticks_per_epoch,
        total_ticks,
        config.activation_kind,
        config.learning_rate,
        world_node_count,
        input_node_count,
        config.weight_seed,
        config.weight_init_scale,
        config.eval_every,
        config
            .checkpoint_every
            .map(|value| value.to_string())
            .unwrap_or_else(|| "disabled".to_owned()),
        config.distributed_workers.len(),
    );

    for epoch_offset in 0..config.epochs {
        let epoch = experiment.starting_epoch() + epoch_offset + 1;
        let base_checkpoint = GraphCheckpoint::from_world(&world, phase)?;
        let reports = dispatch_training_shards(
            &config.distributed_workers,
            &config.train,
            &base_checkpoint,
            &train_dataset,
            config.activation_kind,
            config.learning_rate,
        )?;
        let experiment_goodness = aggregate_experiment_goodness(&reports)?;
        let averaged_checkpoint = average_checkpoints(
            &reports
                .into_iter()
                .map(|report| report.checkpoint)
                .collect::<Vec<_>>(),
        )?;
        let (averaged_world, averaged_phase) = averaged_checkpoint.into_world()?;
        world = averaged_world;
        phase = averaged_phase;

        let summary = summarize_world(&world);
        let evaluation = if epoch % config.eval_every == 0 {
            Some(dispatch_evaluation_shards(
                &config.distributed_workers,
                eval_paths,
                &world,
                phase,
                eval_dataset,
                config.activation_kind,
            )?)
        } else {
            None
        };
        let record = MetricsRecord::from_summaries(
            epoch,
            experiment.elapsed_offset_seconds() + run_started.elapsed().as_secs_f64(),
            summary,
            &experiment_goodness,
            evaluation.as_ref(),
        );
        experiment.append_metrics(&record)?;

        if experiment.should_save_periodic_checkpoint(epoch, final_epoch) {
            let checkpoint_path = experiment.checkpoint_path_for_epoch(epoch);
            save_checkpoint_json(&checkpoint_path, &world, phase)?;
            experiment.record_checkpoint(&checkpoint_path, epoch)?;
            println!("saved checkpoint to {}", checkpoint_path.display());
        }

        if let Some(evaluation) = evaluation {
            println!(
                "epoch {epoch}/{final_epoch} complete | mean_abs_activation={:.6} mean_squared_activation={:.6} mean_abs_weight={:.6} positive_mean_world_goodness={:.6} negative_mean_world_goodness={:.6} goodness_separation={:.6} eval_accuracy={:.6} eval_mean_correct_goodness={:.6} eval_mean_best_wrong_goodness={:.6} eval_mean_margin={:.6}",
                summary.mean_abs_activation,
                summary.mean_squared_activation,
                summary.mean_abs_weight,
                experiment_goodness.positive_mean_world_goodness,
                experiment_goodness.negative_mean_world_goodness,
                experiment_goodness.goodness_separation,
                evaluation.accuracy,
                evaluation.mean_correct_goodness,
                evaluation.mean_best_wrong_goodness,
                evaluation.mean_margin,
            );
        } else {
            println!(
                "epoch {epoch}/{final_epoch} complete | mean_abs_activation={:.6} mean_squared_activation={:.6} mean_abs_weight={:.6} positive_mean_world_goodness={:.6} negative_mean_world_goodness={:.6} goodness_separation={:.6} eval=skipped",
                summary.mean_abs_activation,
                summary.mean_squared_activation,
                summary.mean_abs_weight,
                experiment_goodness.positive_mean_world_goodness,
                experiment_goodness.negative_mean_world_goodness,
                experiment_goodness.goodness_separation,
            );
        }
    }

    if let Some(path) = &config.save_checkpoint {
        save_checkpoint_json(path, &world, phase)?;
        experiment.record_checkpoint(path, final_epoch)?;
        println!("saved final checkpoint to {}", path.display());
    }

    println!(
        "training complete | manifest={} metrics={}",
        experiment.manifest_path().display(),
        experiment.metrics_path().display()
    );

    Ok(TrainingRunOutput {
        run_dir: experiment.root().to_path_buf(),
        manifest_path: experiment.manifest_path().to_path_buf(),
        metrics_path: experiment.metrics_path().to_path_buf(),
        latest_checkpoint_path: experiment
            .latest_checkpoint_path()
            .map(std::path::PathBuf::from),
    })
}

fn ping_workers(workers: &[String]) -> Result<(), DistributedRuntimeError> {
    for worker in workers {
        match request_worker(worker, WorkerRequestPayload::Ping)? {
            WorkerResponsePayload::Pong => {}
            _ => {
                return Err(DistributedRuntimeError::UnexpectedWorkerResponse {
                    worker: worker.clone(),
                });
            }
        }
    }

    Ok(())
}

fn handle_worker_connection(
    mut stream: TcpStream,
    peer: &str,
    dataset_cache: &mut WorkerDatasetCache,
) -> Result<(), DistributedRuntimeError> {
    let request: WorkerRequest = read_message(&mut stream, peer)?;
    let payload = match handle_worker_request(request, dataset_cache) {
        Ok(payload) => payload,
        Err(error) => WorkerResponsePayload::Error(WorkerErrorResponse {
            message: error.to_string(),
        }),
    };
    let response = WorkerResponse {
        protocol_version: DISTRIBUTED_PROTOCOL_VERSION,
        payload,
    };
    write_message(&mut stream, peer, &response)
}

fn handle_worker_request(
    request: WorkerRequest,
    dataset_cache: &mut WorkerDatasetCache,
) -> Result<WorkerResponsePayload, DistributedRuntimeError> {
    if request.protocol_version != DISTRIBUTED_PROTOCOL_VERSION {
        return Err(DistributedRuntimeError::ProtocolVersionMismatch {
            local: DISTRIBUTED_PROTOCOL_VERSION,
            remote: request.protocol_version,
        });
    }

    match request.payload {
        WorkerRequestPayload::Ping => Ok(WorkerResponsePayload::Pong),
        WorkerRequestPayload::TrainShard(request) => {
            let dataset = dataset_cache.load(&request.dataset)?;
            Ok(WorkerResponsePayload::TrainShard(execute_training_shard(
                request.checkpoint,
                dataset,
                request.activation_kind,
                request.learning_rate,
                request.shard,
            )?))
        }
        WorkerRequestPayload::EvaluateShard(request) => {
            let dataset = dataset_cache.load(&request.dataset)?;
            Ok(WorkerResponsePayload::EvaluateShard(
                execute_evaluation_shard(
                    request.checkpoint,
                    dataset,
                    request.activation_kind,
                    request.shard,
                )?,
            ))
        }
    }
}

fn execute_training_shard(
    checkpoint: GraphCheckpoint,
    dataset: &MnistDataset,
    activation_kind: ActivationKind,
    learning_rate: f32,
    shard: ShardRange,
) -> Result<WorkerTrainShardResponse, DistributedRuntimeError> {
    validate_shard(shard, dataset.len())?;

    let (mut world, mut phase) = checkpoint.into_world()?;
    let mut experiment_goodness = ExperimentGoodnessAccumulator::new();
    let mut pixels = [0.0f32; MNIST_IMAGE_PIXELS];

    for sample_index in shard.start..shard.end {
        execute_training_tick(
            &mut world,
            &mut phase,
            dataset,
            sample_index,
            activation_kind,
            learning_rate,
            &mut pixels,
            &mut experiment_goodness,
        )?;
        execute_training_tick(
            &mut world,
            &mut phase,
            dataset,
            sample_index,
            activation_kind,
            learning_rate,
            &mut pixels,
            &mut experiment_goodness,
        )?;
    }

    let experiment_goodness = experiment_goodness.finish()?;
    Ok(WorkerTrainShardResponse {
        checkpoint: GraphCheckpoint::from_world(&world, phase)?,
        experiment_goodness: experiment_goodness.into(),
    })
}

fn execute_training_tick(
    world: &mut World,
    phase: &mut SimulationPhase,
    dataset: &MnistDataset,
    sample_index: usize,
    activation_kind: ActivationKind,
    learning_rate: f32,
    pixels: &mut [f32; MNIST_IMAGE_PIXELS],
    experiment_goodness: &mut ExperimentGoodnessAccumulator,
) -> Result<(), DistributedRuntimeError> {
    let sample_phase = *phase;
    let metadata = match sample_phase {
        SimulationPhase::Positive => fill_positive_sample(dataset, sample_index, pixels)?,
        SimulationPhase::Negative => fill_negative_sample(dataset, sample_index, pixels)?,
    };

    inject_conditioned_sample(world, pixels, metadata.candidate_label)?;
    *phase = phase.toggled();
    update_nodes_forward_forward(world, activation_kind)?;
    experiment_goodness.observe_world(world, sample_phase)?;
    update_local_weights_forward_forward(world, sample_phase, learning_rate)?;

    Ok(())
}

fn execute_evaluation_shard(
    checkpoint: GraphCheckpoint,
    dataset: &MnistDataset,
    activation_kind: ActivationKind,
    shard: ShardRange,
) -> Result<WorkerEvaluateShardResponse, DistributedRuntimeError> {
    validate_shard(shard, dataset.len())?;
    let (world, phase) = checkpoint.into_world()?;
    let accumulator =
        evaluate_forward_forward_range(&world, phase, dataset, activation_kind, shard.range())?;

    Ok(WorkerEvaluateShardResponse { accumulator })
}

fn dispatch_training_shards(
    workers: &[String],
    dataset_paths: &DatasetPaths,
    checkpoint: &GraphCheckpoint,
    dataset: &MnistDataset,
    activation_kind: ActivationKind,
    learning_rate: f32,
) -> Result<Vec<WorkerTrainShardResponse>, DistributedRuntimeError> {
    let assignments = partition_shards(workers, dataset.len())?;
    let mut handles = Vec::with_capacity(assignments.len());

    for (worker, shard) in assignments {
        let request = WorkerTrainShardRequest {
            checkpoint: checkpoint.clone(),
            dataset: dataset_paths.clone(),
            activation_kind,
            learning_rate,
            shard,
        };
        handles.push(thread::spawn(move || {
            let response = request_worker(&worker, WorkerRequestPayload::TrainShard(request))?;
            match response {
                WorkerResponsePayload::TrainShard(response) => Ok(response),
                _ => Err(DistributedRuntimeError::UnexpectedWorkerResponse { worker }),
            }
        }));
    }

    collect_thread_results(handles)
}

fn dispatch_evaluation_shards(
    workers: &[String],
    dataset_paths: &DatasetPaths,
    world: &World,
    phase: SimulationPhase,
    dataset: &MnistDataset,
    activation_kind: ActivationKind,
) -> Result<super::EvaluationSummary, DistributedRuntimeError> {
    let assignments = partition_shards(workers, dataset.len())?;
    let checkpoint = GraphCheckpoint::from_world(world, phase)?;
    let mut handles = Vec::with_capacity(assignments.len());

    for (worker, shard) in assignments {
        let request = WorkerEvaluateShardRequest {
            checkpoint: checkpoint.clone(),
            dataset: dataset_paths.clone(),
            activation_kind,
            shard,
        };
        handles.push(thread::spawn(move || {
            let response = request_worker(&worker, WorkerRequestPayload::EvaluateShard(request))?;
            match response {
                WorkerResponsePayload::EvaluateShard(response) => Ok(response),
                _ => Err(DistributedRuntimeError::UnexpectedWorkerResponse { worker }),
            }
        }));
    }

    let mut accumulator = EvaluationAccumulator::default();
    for response in collect_thread_results(handles)? {
        accumulator.merge(response.accumulator);
    }

    Ok(accumulator.finish())
}

fn collect_thread_results<T>(
    handles: Vec<thread::JoinHandle<Result<T, DistributedRuntimeError>>>,
) -> Result<Vec<T>, DistributedRuntimeError> {
    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        results.push(
            handle
                .join()
                .map_err(|_| DistributedRuntimeError::WorkerThreadPanicked)??,
        );
    }

    Ok(results)
}

fn request_worker(
    worker: &str,
    payload: WorkerRequestPayload,
) -> Result<WorkerResponsePayload, DistributedRuntimeError> {
    let mut stream =
        TcpStream::connect(worker).map_err(|source| DistributedRuntimeError::Transport {
            context: worker.to_owned(),
            source,
        })?;
    let request = WorkerRequest {
        protocol_version: DISTRIBUTED_PROTOCOL_VERSION,
        payload,
    };
    write_message(&mut stream, worker, &request)?;
    let response: WorkerResponse = read_message(&mut stream, worker)?;

    if response.protocol_version != DISTRIBUTED_PROTOCOL_VERSION {
        return Err(DistributedRuntimeError::ProtocolVersionMismatch {
            local: DISTRIBUTED_PROTOCOL_VERSION,
            remote: response.protocol_version,
        });
    }

    match response.payload {
        WorkerResponsePayload::Error(error) => Err(DistributedRuntimeError::WorkerError {
            worker: worker.to_owned(),
            message: error.message,
        }),
        payload => Ok(payload),
    }
}

fn write_message<T: Serialize>(
    stream: &mut TcpStream,
    context: &str,
    message: &T,
) -> Result<(), DistributedRuntimeError> {
    let bytes = serde_json::to_vec(message)?;
    let length =
        u64::try_from(bytes.len()).map_err(|_| DistributedRuntimeError::MessageTooLarge {
            max_bytes: MAX_MESSAGE_BYTES,
        })?;
    if length > MAX_MESSAGE_BYTES as u64 {
        return Err(DistributedRuntimeError::MessageTooLarge {
            max_bytes: MAX_MESSAGE_BYTES,
        });
    }

    stream.write_all(&length.to_be_bytes()).map_err(|source| {
        DistributedRuntimeError::Transport {
            context: context.to_owned(),
            source,
        }
    })?;
    stream
        .write_all(&bytes)
        .map_err(|source| DistributedRuntimeError::Transport {
            context: context.to_owned(),
            source,
        })
}

fn read_message<T: DeserializeOwned>(
    stream: &mut TcpStream,
    context: &str,
) -> Result<T, DistributedRuntimeError> {
    let mut length_bytes = [0u8; 8];
    stream
        .read_exact(&mut length_bytes)
        .map_err(|source| DistributedRuntimeError::Transport {
            context: context.to_owned(),
            source,
        })?;
    let length = u64::from_be_bytes(length_bytes);
    if length > MAX_MESSAGE_BYTES as u64 {
        return Err(DistributedRuntimeError::MessageTooLarge {
            max_bytes: MAX_MESSAGE_BYTES,
        });
    }

    let mut bytes = vec![0u8; length as usize];
    stream
        .read_exact(&mut bytes)
        .map_err(|source| DistributedRuntimeError::Transport {
            context: context.to_owned(),
            source,
        })?;

    Ok(serde_json::from_slice(&bytes)?)
}

fn validate_shard(shard: ShardRange, dataset_len: usize) -> Result<(), DistributedRuntimeError> {
    if shard.start >= shard.end || shard.end > dataset_len {
        return Err(DistributedRuntimeError::InvalidShard {
            start: shard.start,
            end: shard.end,
            dataset_len,
        });
    }

    Ok(())
}

fn partition_shards(
    workers: &[String],
    sample_count: usize,
) -> Result<Vec<(String, ShardRange)>, DistributedRuntimeError> {
    if workers.is_empty() {
        return Err(DistributedRuntimeError::NoWorkersConfigured);
    }
    if sample_count == 0 {
        return Err(DistributedRuntimeError::NoWorkerResults);
    }

    let shard_count = workers.len().min(sample_count);
    let base = sample_count / shard_count;
    let remainder = sample_count % shard_count;
    let mut start = 0usize;
    let mut shards = Vec::with_capacity(shard_count);

    for (index, worker) in workers.iter().take(shard_count).enumerate() {
        let shard_len = base + usize::from(index < remainder);
        let end = start + shard_len;
        shards.push((worker.clone(), ShardRange { start, end }));
        start = end;
    }

    Ok(shards)
}

fn aggregate_experiment_goodness(
    reports: &[WorkerTrainShardResponse],
) -> Result<ExperimentGoodness, DistributedRuntimeError> {
    if reports.is_empty() {
        return Err(DistributedRuntimeError::NoWorkerResults);
    }

    let positive_samples = reports
        .iter()
        .map(|report| report.experiment_goodness.positive_samples)
        .sum::<usize>();
    let negative_samples = reports
        .iter()
        .map(|report| report.experiment_goodness.negative_samples)
        .sum::<usize>();

    if positive_samples == 0 || negative_samples == 0 {
        return Err(DistributedRuntimeError::NoWorkerResults);
    }

    let positive_sum = reports
        .iter()
        .map(|report| {
            report.experiment_goodness.positive_mean_world_goodness
                * report.experiment_goodness.positive_samples as f32
        })
        .sum::<f32>();
    let negative_sum = reports
        .iter()
        .map(|report| {
            report.experiment_goodness.negative_mean_world_goodness
                * report.experiment_goodness.negative_samples as f32
        })
        .sum::<f32>();
    let positive_mean_world_goodness = positive_sum / positive_samples as f32;
    let negative_mean_world_goodness = negative_sum / negative_samples as f32;

    Ok(ExperimentGoodness {
        positive_mean_world_goodness,
        negative_mean_world_goodness,
        goodness_separation: positive_mean_world_goodness - negative_mean_world_goodness,
        positive_samples,
        negative_samples,
    })
}

fn average_checkpoints(
    checkpoints: &[GraphCheckpoint],
) -> Result<GraphCheckpoint, DistributedRuntimeError> {
    let mut averaged = checkpoints
        .first()
        .cloned()
        .ok_or(DistributedRuntimeError::NoWorkerResults)?;
    let worker_count = checkpoints.len() as f32;

    for checkpoint in checkpoints.iter().skip(1) {
        if checkpoint.format_version != averaged.format_version {
            return Err(DistributedRuntimeError::IncompatibleCheckpoint {
                node: 0,
                reason: "format version mismatch",
            });
        }
        if checkpoint.phase != averaged.phase {
            return Err(DistributedRuntimeError::IncompatibleCheckpoint {
                node: 0,
                reason: "simulation phase mismatch",
            });
        }
        if checkpoint.nodes.len() != averaged.nodes.len() {
            return Err(DistributedRuntimeError::IncompatibleCheckpoint {
                node: 0,
                reason: "node count mismatch",
            });
        }

        for (node_index, (left, right)) in averaged
            .nodes
            .iter_mut()
            .zip(checkpoint.nodes.iter())
            .enumerate()
        {
            ensure_checkpoint_node_compatible(node_index, left, right)?;
            left.activation += right.activation;
            for slot in 0..left.local_weights.neighbor_weights.len() {
                left.local_weights.neighbor_weights[slot] +=
                    right.local_weights.neighbor_weights[slot];
            }
        }
    }

    for node in &mut averaged.nodes {
        node.activation /= worker_count;
        for weight in &mut node.local_weights.neighbor_weights {
            *weight /= worker_count;
        }
    }

    Ok(averaged)
}

fn ensure_checkpoint_node_compatible(
    node_index: usize,
    left: &GraphNodeCheckpoint,
    right: &GraphNodeCheckpoint,
) -> Result<(), DistributedRuntimeError> {
    if left.index != right.index {
        return Err(DistributedRuntimeError::IncompatibleCheckpoint {
            node: node_index,
            reason: "stable node index mismatch",
        });
    }
    if left.is_input != right.is_input {
        return Err(DistributedRuntimeError::IncompatibleCheckpoint {
            node: node_index,
            reason: "input-node marker mismatch",
        });
    }
    if left.topology != right.topology {
        return Err(DistributedRuntimeError::IncompatibleCheckpoint {
            node: node_index,
            reason: "topology mismatch",
        });
    }

    Ok(())
}

impl ShardRange {
    fn range(self) -> Range<usize> {
        self.start..self.end
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs_runtime::checkpoint::StableTopologyPointers;
    use crate::ecs_runtime::components::LocalWeights;
    use crate::ecs_runtime::ingestion_system::SimulationPhase;

    #[test]
    fn partitions_samples_without_empty_shards() {
        let shards = partition_shards(
            &[
                "worker-a:7000".to_owned(),
                "worker-b:7000".to_owned(),
                "worker-c:7000".to_owned(),
            ],
            5,
        )
        .expect("partition shards");

        assert_eq!(shards.len(), 3);
        assert_eq!(shards[0].1, ShardRange { start: 0, end: 2 });
        assert_eq!(shards[1].1, ShardRange { start: 2, end: 4 });
        assert_eq!(shards[2].1, ShardRange { start: 4, end: 5 });
    }

    #[test]
    fn averages_worker_checkpoints_elementwise() {
        let left = GraphCheckpoint {
            format_version: 1,
            phase: SimulationPhase::Positive,
            nodes: vec![GraphNodeCheckpoint {
                index: 0,
                activation: 2.0,
                is_input: true,
                local_weights: LocalWeights::new([1.0, 3.0, 5.0]),
                topology: StableTopologyPointers::new([0, 0, 0]),
            }],
        };
        let right = GraphCheckpoint {
            format_version: 1,
            phase: SimulationPhase::Positive,
            nodes: vec![GraphNodeCheckpoint {
                index: 0,
                activation: 4.0,
                is_input: true,
                local_weights: LocalWeights::new([3.0, 5.0, 7.0]),
                topology: StableTopologyPointers::new([0, 0, 0]),
            }],
        };

        let averaged = average_checkpoints(&[left, right]).expect("average checkpoints");

        assert_eq!(averaged.nodes[0].activation, 3.0);
        assert_eq!(
            averaged.nodes[0].local_weights.neighbor_weights,
            [2.0, 4.0, 6.0]
        );
    }

    #[test]
    fn aggregates_weighted_worker_goodness() {
        let aggregated = aggregate_experiment_goodness(&[
            WorkerTrainShardResponse {
                checkpoint: sample_checkpoint(),
                experiment_goodness: ShardExperimentMetrics {
                    positive_mean_world_goodness: 2.0,
                    negative_mean_world_goodness: 1.0,
                    positive_samples: 2,
                    negative_samples: 2,
                },
            },
            WorkerTrainShardResponse {
                checkpoint: sample_checkpoint(),
                experiment_goodness: ShardExperimentMetrics {
                    positive_mean_world_goodness: 4.0,
                    negative_mean_world_goodness: 3.0,
                    positive_samples: 1,
                    negative_samples: 1,
                },
            },
        ])
        .expect("aggregate experiment goodness");

        assert!((aggregated.positive_mean_world_goodness - (8.0 / 3.0)).abs() < 1.0e-6);
        assert!((aggregated.negative_mean_world_goodness - (5.0 / 3.0)).abs() < 1.0e-6);
        assert!((aggregated.goodness_separation - 1.0).abs() < 1.0e-6);
    }

    fn sample_checkpoint() -> GraphCheckpoint {
        GraphCheckpoint {
            format_version: 1,
            phase: SimulationPhase::Positive,
            nodes: vec![GraphNodeCheckpoint {
                index: 0,
                activation: 0.0,
                is_input: true,
                local_weights: LocalWeights::filled(0.0),
                topology: StableTopologyPointers::new([0, 0, 0]),
            }],
        }
    }
}
