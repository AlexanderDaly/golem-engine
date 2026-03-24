use std::collections::HashMap;
use std::io::{self, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::ops::Range;
use std::thread;
use std::time::Instant;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::data::mnist_loader::{MnistDataset, MNIST_IMAGE_PIXELS};
use crate::ecs_runtime::checkpoint::{
    save_graph_checkpoint_json, CheckpointError, GraphCheckpoint, GraphNodeCheckpoint,
};
use crate::ecs_runtime::ingestion_system::{
    fill_positive_sample, ContrastiveDataStream, IngestionError, SimulationPhase,
};
use crate::ecs_runtime::systems::ActivationKind;
use crate::experiment::{node_local_goodness, ExperimentGoodness};
use crate::{CONDITIONED_INPUT_NODE_COUNT, MNIST_LABEL_CLASS_COUNT, REGULAR_DEGREE};

use super::cli::WorkerServerConfig;
use super::eval::EvaluationSummary;
use super::world::WorldSummary;
use super::{
    EffectiveRunSettings, ExperimentRun, MetricsRecord, PreparedTrainingState, TrainingConfig,
    TrainingRunError, TrainingRunOutput,
};

const DISTRIBUTED_PROTOCOL_VERSION: u32 = 2;
const MAX_MESSAGE_BYTES: usize = 512 * 1024 * 1024;
const LABEL_SLOT_OFFSET: usize = MNIST_IMAGE_PIXELS;

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
    #[error("worker shard state has not been initialized")]
    WorkerNotInitialized,
    #[error("worker shard state has already been initialized for this connection")]
    WorkerAlreadyInitialized,
    #[error("worker shard contains duplicate node {node}")]
    DuplicateOwnedNode { node: usize },
    #[error("worker shard does not own node {node}")]
    UnknownOwnedNode { node: usize },
    #[error("failed to partition {node_count} nodes across {worker_count} workers")]
    InvalidNodePartition {
        node_count: usize,
        worker_count: usize,
    },
    #[error(
        "label-conditioned Forward-Forward requires at least {required} input nodes; found {found}"
    )]
    InsufficientInputNodes { found: usize, required: usize },
    #[error("distributed protocol message exceeds the {max_bytes}-byte safety limit")]
    MessageTooLarge { max_bytes: usize },
    #[error("checkpoint operation failed during distributed execution: {0}")]
    Checkpoint(#[from] CheckpointError),
    #[error("data ingestion failed during distributed execution: {0}")]
    Ingestion(#[from] IngestionError),
    #[error("distributed experiment goodness is missing positive observations")]
    MissingPositiveSamples,
    #[error("distributed experiment goodness is missing negative observations")]
    MissingNegativeSamples,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerRequest {
    protocol_version: u32,
    payload: WorkerRequestPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum WorkerRequestPayload {
    Ping,
    InitializeShard(InitializeShardRequest),
    SetNodeActivations(SetNodeActivationsRequest),
    ResetActivations,
    ForwardNode(ForwardNodeRequest),
    UpdateWeightsBatch(UpdateWeightsBatchRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerResponse {
    protocol_version: u32,
    payload: WorkerResponsePayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum WorkerResponsePayload {
    Pong,
    Ack,
    ForwardNode(ForwardNodeResponse),
    UpdateWeightsBatch(UpdateWeightsBatchResponse),
    Error(WorkerErrorResponse),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerErrorResponse {
    message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InitializeShardRequest {
    nodes: Vec<GraphNodeCheckpoint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct NodeActivationPatch {
    node_index: usize,
    activation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SetNodeActivationsRequest {
    patches: Vec<NodeActivationPatch>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct ForwardNodeRequest {
    node_index: usize,
    neighbor_activations: [f32; REGULAR_DEGREE],
    activation_kind: ActivationKind,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct ForwardNodeResponse {
    activation: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct WeightUpdateNodeRequest {
    node_index: usize,
    neighbor_activations: [f32; REGULAR_DEGREE],
    phase: SimulationPhase,
    learning_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UpdateWeightsBatchRequest {
    updates: Vec<WeightUpdateNodeRequest>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct WeightUpdateNodeResponse {
    node_index: usize,
    neighbor_weights: [f32; REGULAR_DEGREE],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UpdateWeightsBatchResponse {
    updates: Vec<WeightUpdateNodeResponse>,
}

struct WorkerShardState {
    nodes: HashMap<usize, GraphNodeCheckpoint>,
}

impl WorkerShardState {
    fn initialize(nodes: Vec<GraphNodeCheckpoint>) -> Result<Self, DistributedRuntimeError> {
        let mut owned_nodes = HashMap::with_capacity(nodes.len());
        for node in nodes {
            let node_index = node.index;
            if owned_nodes.insert(node_index, node).is_some() {
                return Err(DistributedRuntimeError::DuplicateOwnedNode { node: node_index });
            }
        }

        Ok(Self { nodes: owned_nodes })
    }

    fn set_node_activations(
        &mut self,
        patches: &[NodeActivationPatch],
    ) -> Result<(), DistributedRuntimeError> {
        for patch in patches {
            self.node_mut(patch.node_index)?.activation = patch.activation;
        }

        Ok(())
    }

    fn reset_activations(&mut self) {
        for node in self.nodes.values_mut() {
            node.activation = 0.0;
        }
    }

    fn forward_node(
        &mut self,
        request: ForwardNodeRequest,
    ) -> Result<ForwardNodeResponse, DistributedRuntimeError> {
        let node = self.node_mut(request.node_index)?;
        let local_input_sum: f32 = node
            .local_weights
            .neighbor_weights
            .iter()
            .zip(request.neighbor_activations.iter())
            .map(|(weight, activation)| weight * activation)
            .sum();
        let activation = request.activation_kind.apply(local_input_sum);
        node.activation = activation;

        Ok(ForwardNodeResponse { activation })
    }

    fn update_weights_batch(
        &mut self,
        request: UpdateWeightsBatchRequest,
    ) -> Result<UpdateWeightsBatchResponse, DistributedRuntimeError> {
        let mut updates = Vec::with_capacity(request.updates.len());

        for update in request.updates {
            let node = self.node_mut(update.node_index)?;
            let local_activation = node.activation;
            let goodness = node_local_goodness(local_activation, &update.neighbor_activations);

            for (slot, neighbor_activation) in
                update.neighbor_activations.iter().copied().enumerate()
            {
                let local_correlation = local_activation * neighbor_activation;
                let delta = update.phase.learning_direction()
                    * update.learning_rate
                    * goodness
                    * local_correlation;
                node.local_weights.neighbor_weights[slot] += delta;
            }

            updates.push(WeightUpdateNodeResponse {
                node_index: update.node_index,
                neighbor_weights: node.local_weights.neighbor_weights,
            });
        }

        Ok(UpdateWeightsBatchResponse { updates })
    }

    fn node_mut(
        &mut self,
        node_index: usize,
    ) -> Result<&mut GraphNodeCheckpoint, DistributedRuntimeError> {
        self.nodes
            .get_mut(&node_index)
            .ok_or(DistributedRuntimeError::UnknownOwnedNode { node: node_index })
    }
}

pub fn run_worker_server(config: WorkerServerConfig) -> Result<(), DistributedRuntimeError> {
    let listener =
        TcpListener::bind(&config.listen_addr).map_err(|source| DistributedRuntimeError::Bind {
            addr: config.listen_addr.clone(),
            source,
        })?;

    println!("distributed worker listening on {}", config.listen_addr);

    loop {
        let (stream, peer_addr) =
            listener
                .accept()
                .map_err(|source| DistributedRuntimeError::Accept {
                    addr: config.listen_addr.clone(),
                    source,
                })?;
        thread::spawn(move || {
            if let Err(error) = handle_worker_connection(stream, &peer_addr.to_string()) {
                eprintln!("worker request failed: {error}");
            }
        });
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
        world,
        phase,
        effective_graph_search_limit,
        world_node_count,
        input_node_count,
    } = prepared;

    let mut runtime = DistributedCheckpointRuntime::connect(
        &config.distributed_workers,
        GraphCheckpoint::from_world(&world, phase)?,
    )?;

    let mut experiment = ExperimentRun::prepare(
        &config,
        EffectiveRunSettings {
            graph_node_count: world_node_count,
            effective_graph_search_limit,
        },
    )?;
    let final_epoch = experiment.starting_epoch() + config.epochs;
    let eval_dataset = test_dataset.as_ref().unwrap_or(&train_dataset);
    let mut stream = ContrastiveDataStream::new(&train_dataset)?;
    let ticks_per_epoch =
        stream
            .dataset()
            .len()
            .checked_mul(2)
            .ok_or(TrainingRunError::TickCountOverflow {
                dataset_len: stream.dataset().len(),
                epochs: config.epochs,
            })?;
    let total_ticks =
        ticks_per_epoch
            .checked_mul(config.epochs)
            .ok_or(TrainingRunError::TickCountOverflow {
                dataset_len: stream.dataset().len(),
                epochs: config.epochs,
            })?;
    let run_started = Instant::now();
    let mut pixels = [0.0f32; MNIST_IMAGE_PIXELS];

    println!("run directory: {}", experiment.root().display());
    println!(
        "starting distributed training: start_epoch={} end_epoch={} epochs_this_invocation={} ticks_per_epoch={} total_ticks={} activation={} learning_rate={} graph_node_count={} input_node_count={} weight_seed={} weight_init_scale={} eval_every={} checkpoint_every={} workers={} strategy=partitioned_world",
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
        runtime.worker_count(),
    );

    for epoch_offset in 0..config.epochs {
        let epoch = experiment.starting_epoch() + epoch_offset + 1;
        let mut experiment_goodness = PhaseGoodnessAccumulator::default();

        for _tick in 0..ticks_per_epoch {
            let sample_phase = runtime.phase();
            let metadata = match sample_phase {
                SimulationPhase::Positive => stream.fill_next_positive(&mut pixels)?,
                SimulationPhase::Negative => stream.fill_next_negative(&mut pixels)?,
            };

            runtime.apply_conditioned_sample(&pixels, metadata.candidate_label)?;
            runtime.set_phase(sample_phase.toggled());
            runtime.forward_sweep(config.activation_kind)?;
            experiment_goodness.observe(sample_phase, runtime.mean_world_goodness());
            runtime.update_weights(sample_phase, config.learning_rate)?;
        }

        let summary = runtime.summary();
        let experiment_goodness = experiment_goodness.finish()?;
        let evaluation = if epoch % config.eval_every == 0 {
            Some(evaluate_partitioned_world(
                &config.distributed_workers,
                runtime.checkpoint(),
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
            save_graph_checkpoint_json(&checkpoint_path, runtime.checkpoint())?;
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
        save_graph_checkpoint_json(path, runtime.checkpoint())?;
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

struct DistributedCheckpointRuntime {
    checkpoint: GraphCheckpoint,
    workers: Vec<PartitionedWorkerClient>,
    owner_by_node: Vec<usize>,
    input_node_indices: Vec<usize>,
}

impl DistributedCheckpointRuntime {
    fn connect(
        worker_addresses: &[String],
        checkpoint: GraphCheckpoint,
    ) -> Result<Self, DistributedRuntimeError> {
        let assignments = partition_node_ranges(worker_addresses, checkpoint.nodes.len())?;
        let mut workers = Vec::with_capacity(assignments.len());
        let mut owner_by_node = vec![0usize; checkpoint.nodes.len()];

        for (worker_index, (worker, range)) in assignments.into_iter().enumerate() {
            for node_index in range.clone() {
                owner_by_node[node_index] = worker_index;
            }
            let shard_nodes = checkpoint.nodes[range.clone()].to_vec();
            workers.push(PartitionedWorkerClient::connect(
                worker,
                range,
                shard_nodes,
            )?);
        }

        let input_node_indices: Vec<usize> = checkpoint
            .nodes
            .iter()
            .filter(|node| node.is_input)
            .map(|node| node.index)
            .collect();
        if input_node_indices.len() < CONDITIONED_INPUT_NODE_COUNT {
            return Err(DistributedRuntimeError::InsufficientInputNodes {
                found: input_node_indices.len(),
                required: CONDITIONED_INPUT_NODE_COUNT,
            });
        }

        Ok(Self {
            checkpoint,
            workers,
            owner_by_node,
            input_node_indices,
        })
    }

    fn worker_count(&self) -> usize {
        self.workers.len()
    }

    fn checkpoint(&self) -> &GraphCheckpoint {
        &self.checkpoint
    }

    fn phase(&self) -> SimulationPhase {
        self.checkpoint.phase
    }

    fn set_phase(&mut self, phase: SimulationPhase) {
        self.checkpoint.phase = phase;
    }

    fn apply_conditioned_sample(
        &mut self,
        pixels: &[f32; MNIST_IMAGE_PIXELS],
        candidate_label: u8,
    ) -> Result<(), DistributedRuntimeError> {
        let mut patches_by_worker = vec![Vec::new(); self.workers.len()];

        for (slot, node_index) in self.input_node_indices.iter().copied().enumerate() {
            let activation = conditioned_slot_activation(pixels, slot, candidate_label);
            self.checkpoint.nodes[node_index].activation = activation;
            patches_by_worker[self.owner_by_node[node_index]].push(NodeActivationPatch {
                node_index,
                activation,
            });
        }

        for (worker_index, patches) in patches_by_worker.into_iter().enumerate() {
            if !patches.is_empty() {
                self.workers[worker_index].set_node_activations(patches)?;
            }
        }

        Ok(())
    }

    fn reset_activations(&mut self) -> Result<(), DistributedRuntimeError> {
        for node in &mut self.checkpoint.nodes {
            node.activation = 0.0;
        }

        for worker in &mut self.workers {
            worker.reset_activations()?;
        }

        Ok(())
    }

    fn forward_sweep(
        &mut self,
        activation_kind: ActivationKind,
    ) -> Result<(), DistributedRuntimeError> {
        for node_index in 0..self.checkpoint.nodes.len() {
            let neighbor_activations = neighbor_activations(&self.checkpoint, node_index);
            let owner = self.owner_by_node[node_index];
            let activation = self.workers[owner].forward_node(
                node_index,
                neighbor_activations,
                activation_kind,
            )?;
            self.checkpoint.nodes[node_index].activation = activation;
        }

        Ok(())
    }

    fn update_weights(
        &mut self,
        phase: SimulationPhase,
        learning_rate: f32,
    ) -> Result<(), DistributedRuntimeError> {
        let mut updates_by_worker = vec![Vec::new(); self.workers.len()];

        for node_index in 0..self.checkpoint.nodes.len() {
            updates_by_worker[self.owner_by_node[node_index]].push(WeightUpdateNodeRequest {
                node_index,
                neighbor_activations: neighbor_activations(&self.checkpoint, node_index),
                phase,
                learning_rate,
            });
        }

        for (worker_index, updates) in updates_by_worker.into_iter().enumerate() {
            if updates.is_empty() {
                continue;
            }

            let response = self.workers[worker_index].update_weights_batch(updates)?;
            for update in response.updates {
                self.checkpoint.nodes[update.node_index]
                    .local_weights
                    .neighbor_weights = update.neighbor_weights;
            }
        }

        Ok(())
    }

    fn mean_world_goodness(&self) -> f32 {
        mean_world_goodness_from_checkpoint(&self.checkpoint)
    }

    fn summary(&self) -> WorldSummary {
        summarize_checkpoint(&self.checkpoint)
    }
}

struct PartitionedWorkerClient {
    worker: String,
    node_range: Range<usize>,
    stream: TcpStream,
}

impl PartitionedWorkerClient {
    fn connect(
        worker: String,
        node_range: Range<usize>,
        shard_nodes: Vec<GraphNodeCheckpoint>,
    ) -> Result<Self, DistributedRuntimeError> {
        let stream =
            TcpStream::connect(&worker).map_err(|source| DistributedRuntimeError::Transport {
                context: worker.clone(),
                source,
            })?;
        stream
            .set_nodelay(true)
            .map_err(|source| DistributedRuntimeError::Transport {
                context: worker.clone(),
                source,
            })?;

        let mut client = Self {
            worker,
            node_range,
            stream,
        };

        match client.request(WorkerRequestPayload::Ping)? {
            WorkerResponsePayload::Pong => {}
            _ => {
                return Err(DistributedRuntimeError::UnexpectedWorkerResponse {
                    worker: client.worker.clone(),
                });
            }
        }

        client.initialize_shard(shard_nodes)?;
        Ok(client)
    }

    fn initialize_shard(
        &mut self,
        shard_nodes: Vec<GraphNodeCheckpoint>,
    ) -> Result<(), DistributedRuntimeError> {
        match self.request(WorkerRequestPayload::InitializeShard(
            InitializeShardRequest { nodes: shard_nodes },
        ))? {
            WorkerResponsePayload::Ack => Ok(()),
            _ => Err(DistributedRuntimeError::UnexpectedWorkerResponse {
                worker: self.worker.clone(),
            }),
        }
    }

    fn set_node_activations(
        &mut self,
        patches: Vec<NodeActivationPatch>,
    ) -> Result<(), DistributedRuntimeError> {
        match self.request(WorkerRequestPayload::SetNodeActivations(
            SetNodeActivationsRequest { patches },
        ))? {
            WorkerResponsePayload::Ack => Ok(()),
            _ => Err(DistributedRuntimeError::UnexpectedWorkerResponse {
                worker: self.worker.clone(),
            }),
        }
    }

    fn reset_activations(&mut self) -> Result<(), DistributedRuntimeError> {
        match self.request(WorkerRequestPayload::ResetActivations)? {
            WorkerResponsePayload::Ack => Ok(()),
            _ => Err(DistributedRuntimeError::UnexpectedWorkerResponse {
                worker: self.worker.clone(),
            }),
        }
    }

    fn forward_node(
        &mut self,
        node_index: usize,
        neighbor_activations: [f32; REGULAR_DEGREE],
        activation_kind: ActivationKind,
    ) -> Result<f32, DistributedRuntimeError> {
        debug_assert!(self.node_range.contains(&node_index));

        match self.request(WorkerRequestPayload::ForwardNode(ForwardNodeRequest {
            node_index,
            neighbor_activations,
            activation_kind,
        }))? {
            WorkerResponsePayload::ForwardNode(response) => Ok(response.activation),
            _ => Err(DistributedRuntimeError::UnexpectedWorkerResponse {
                worker: self.worker.clone(),
            }),
        }
    }

    fn update_weights_batch(
        &mut self,
        updates: Vec<WeightUpdateNodeRequest>,
    ) -> Result<UpdateWeightsBatchResponse, DistributedRuntimeError> {
        match self.request(WorkerRequestPayload::UpdateWeightsBatch(
            UpdateWeightsBatchRequest { updates },
        ))? {
            WorkerResponsePayload::UpdateWeightsBatch(response) => Ok(response),
            _ => Err(DistributedRuntimeError::UnexpectedWorkerResponse {
                worker: self.worker.clone(),
            }),
        }
    }

    fn request(
        &mut self,
        payload: WorkerRequestPayload,
    ) -> Result<WorkerResponsePayload, DistributedRuntimeError> {
        let request = WorkerRequest {
            protocol_version: DISTRIBUTED_PROTOCOL_VERSION,
            payload,
        };
        write_message(&mut self.stream, &self.worker, &request)?;
        let response: WorkerResponse = read_message(&mut self.stream, &self.worker)?;

        if response.protocol_version != DISTRIBUTED_PROTOCOL_VERSION {
            return Err(DistributedRuntimeError::ProtocolVersionMismatch {
                local: DISTRIBUTED_PROTOCOL_VERSION,
                remote: response.protocol_version,
            });
        }

        match response.payload {
            WorkerResponsePayload::Error(error) => Err(DistributedRuntimeError::WorkerError {
                worker: self.worker.clone(),
                message: error.message,
            }),
            payload => Ok(payload),
        }
    }
}

fn handle_worker_connection(
    mut stream: TcpStream,
    peer: &str,
) -> Result<(), DistributedRuntimeError> {
    let mut shard_state: Option<WorkerShardState> = None;

    loop {
        let request: WorkerRequest = match try_read_message(&mut stream, peer)? {
            Some(request) => request,
            None => return Ok(()),
        };
        let payload = match handle_worker_request(request, &mut shard_state) {
            Ok(payload) => payload,
            Err(error) => WorkerResponsePayload::Error(WorkerErrorResponse {
                message: error.to_string(),
            }),
        };
        let response = WorkerResponse {
            protocol_version: DISTRIBUTED_PROTOCOL_VERSION,
            payload,
        };
        write_message(&mut stream, peer, &response)?;
    }
}

fn handle_worker_request(
    request: WorkerRequest,
    shard_state: &mut Option<WorkerShardState>,
) -> Result<WorkerResponsePayload, DistributedRuntimeError> {
    if request.protocol_version != DISTRIBUTED_PROTOCOL_VERSION {
        return Err(DistributedRuntimeError::ProtocolVersionMismatch {
            local: DISTRIBUTED_PROTOCOL_VERSION,
            remote: request.protocol_version,
        });
    }

    match request.payload {
        WorkerRequestPayload::Ping => Ok(WorkerResponsePayload::Pong),
        WorkerRequestPayload::InitializeShard(request) => {
            if shard_state.is_some() {
                return Err(DistributedRuntimeError::WorkerAlreadyInitialized);
            }

            *shard_state = Some(WorkerShardState::initialize(request.nodes)?);
            Ok(WorkerResponsePayload::Ack)
        }
        WorkerRequestPayload::SetNodeActivations(request) => {
            shard_state
                .as_mut()
                .ok_or(DistributedRuntimeError::WorkerNotInitialized)?
                .set_node_activations(&request.patches)?;
            Ok(WorkerResponsePayload::Ack)
        }
        WorkerRequestPayload::ResetActivations => {
            shard_state
                .as_mut()
                .ok_or(DistributedRuntimeError::WorkerNotInitialized)?
                .reset_activations();
            Ok(WorkerResponsePayload::Ack)
        }
        WorkerRequestPayload::ForwardNode(request) => Ok(WorkerResponsePayload::ForwardNode(
            shard_state
                .as_mut()
                .ok_or(DistributedRuntimeError::WorkerNotInitialized)?
                .forward_node(request)?,
        )),
        WorkerRequestPayload::UpdateWeightsBatch(request) => {
            Ok(WorkerResponsePayload::UpdateWeightsBatch(
                shard_state
                    .as_mut()
                    .ok_or(DistributedRuntimeError::WorkerNotInitialized)?
                    .update_weights_batch(request)?,
            ))
        }
    }
}

fn evaluate_partitioned_world(
    worker_addresses: &[String],
    checkpoint: &GraphCheckpoint,
    dataset: &MnistDataset,
    activation_kind: ActivationKind,
) -> Result<EvaluationSummary, DistributedRuntimeError> {
    let mut runtime = DistributedCheckpointRuntime::connect(worker_addresses, checkpoint.clone())?;
    let mut pixels = [0.0f32; MNIST_IMAGE_PIXELS];
    let mut correct_predictions = 0usize;
    let mut correct_goodness_sum = 0.0f32;
    let mut best_wrong_goodness_sum = 0.0f32;
    let mut margin_sum = 0.0f32;

    for index in 0..dataset.len() {
        let sample = fill_positive_sample(dataset, index, &mut pixels)?;
        let evaluation =
            evaluate_conditioned_pixels(&mut runtime, &pixels, sample.true_label, activation_kind)?;

        correct_predictions += usize::from(evaluation.predicted_label == sample.true_label);
        correct_goodness_sum += evaluation.correct_goodness;
        best_wrong_goodness_sum += evaluation.best_wrong_goodness;
        margin_sum += evaluation.margin();
    }

    let sample_count = dataset.len() as f32;
    Ok(EvaluationSummary {
        accuracy: correct_predictions as f32 / sample_count,
        mean_correct_goodness: correct_goodness_sum / sample_count,
        mean_best_wrong_goodness: best_wrong_goodness_sum / sample_count,
        mean_margin: margin_sum / sample_count,
    })
}

fn evaluate_conditioned_pixels(
    runtime: &mut DistributedCheckpointRuntime,
    pixels: &[f32; MNIST_IMAGE_PIXELS],
    true_label: u8,
    activation_kind: ActivationKind,
) -> Result<DistributedSampleEvaluation, DistributedRuntimeError> {
    let mut predicted_label = 0u8;
    let mut predicted_goodness = f32::NEG_INFINITY;
    let mut correct_goodness = f32::NEG_INFINITY;
    let mut best_wrong_goodness = f32::NEG_INFINITY;

    for candidate_label in 0..MNIST_LABEL_CLASS_COUNT as u8 {
        runtime.reset_activations()?;
        runtime.apply_conditioned_sample(pixels, candidate_label)?;
        runtime.forward_sweep(activation_kind)?;
        let goodness = runtime.mean_world_goodness();

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

    Ok(DistributedSampleEvaluation {
        predicted_label,
        correct_goodness,
        best_wrong_goodness,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct DistributedSampleEvaluation {
    predicted_label: u8,
    correct_goodness: f32,
    best_wrong_goodness: f32,
}

impl DistributedSampleEvaluation {
    fn margin(self) -> f32 {
        self.correct_goodness - self.best_wrong_goodness
    }
}

#[derive(Default)]
struct PhaseGoodnessAccumulator {
    positive_goodness_sum: f32,
    negative_goodness_sum: f32,
    positive_samples: usize,
    negative_samples: usize,
}

impl PhaseGoodnessAccumulator {
    fn observe(&mut self, phase: SimulationPhase, mean_world_goodness: f32) {
        match phase {
            SimulationPhase::Positive => {
                self.positive_goodness_sum += mean_world_goodness;
                self.positive_samples += 1;
            }
            SimulationPhase::Negative => {
                self.negative_goodness_sum += mean_world_goodness;
                self.negative_samples += 1;
            }
        }
    }

    fn finish(self) -> Result<ExperimentGoodness, DistributedRuntimeError> {
        if self.positive_samples == 0 {
            return Err(DistributedRuntimeError::MissingPositiveSamples);
        }
        if self.negative_samples == 0 {
            return Err(DistributedRuntimeError::MissingNegativeSamples);
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

fn neighbor_activations(checkpoint: &GraphCheckpoint, node_index: usize) -> [f32; REGULAR_DEGREE] {
    std::array::from_fn(|slot| {
        let neighbor_index = checkpoint.nodes[node_index].topology.neighbors[slot];
        checkpoint.nodes[neighbor_index].activation
    })
}

fn mean_world_goodness_from_checkpoint(checkpoint: &GraphCheckpoint) -> f32 {
    let goodness_sum = checkpoint
        .nodes
        .iter()
        .map(|node| {
            node_local_goodness(
                node.activation,
                &std::array::from_fn(|slot| {
                    checkpoint.nodes[node.topology.neighbors[slot]].activation
                }),
            )
        })
        .sum::<f32>();

    goodness_sum / checkpoint.nodes.len() as f32
}

fn summarize_checkpoint(checkpoint: &GraphCheckpoint) -> WorldSummary {
    let activation_count = checkpoint.nodes.len() as f32;
    let weight_count = (checkpoint.nodes.len() * REGULAR_DEGREE) as f32;
    let abs_activation_sum = checkpoint
        .nodes
        .iter()
        .map(|node| node.activation.abs())
        .sum::<f32>();
    let squared_activation_sum = checkpoint
        .nodes
        .iter()
        .map(|node| node.activation * node.activation)
        .sum::<f32>();
    let abs_weight_sum = checkpoint
        .nodes
        .iter()
        .flat_map(|node| node.local_weights.neighbor_weights)
        .map(f32::abs)
        .sum::<f32>();

    WorldSummary {
        mean_abs_activation: abs_activation_sum / activation_count,
        mean_squared_activation: squared_activation_sum / activation_count,
        mean_abs_weight: abs_weight_sum / weight_count,
    }
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
        return if slot - LABEL_SLOT_OFFSET == usize::from(candidate_label) {
            1.0
        } else {
            0.0
        };
    }

    0.0
}

fn partition_node_ranges(
    workers: &[String],
    node_count: usize,
) -> Result<Vec<(String, Range<usize>)>, DistributedRuntimeError> {
    if workers.is_empty() {
        return Err(DistributedRuntimeError::NoWorkersConfigured);
    }
    if node_count == 0 {
        return Err(DistributedRuntimeError::InvalidNodePartition {
            node_count,
            worker_count: workers.len(),
        });
    }

    let shard_count = workers.len().min(node_count);
    let base = node_count / shard_count;
    let remainder = node_count % shard_count;
    let mut start = 0usize;
    let mut assignments = Vec::with_capacity(shard_count);

    for (index, worker) in workers.iter().take(shard_count).enumerate() {
        let shard_len = base + usize::from(index < remainder);
        let end = start + shard_len;
        assignments.push((worker.clone(), start..end));
        start = end;
    }

    Ok(assignments)
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

fn try_read_message<T: DeserializeOwned>(
    stream: &mut TcpStream,
    context: &str,
) -> Result<Option<T>, DistributedRuntimeError> {
    let mut length_bytes = [0u8; 8];
    match stream.read(&mut length_bytes[..1]) {
        Ok(0) => return Ok(None),
        Ok(1) => {}
        Ok(_) => unreachable!("single-byte read should produce at most one byte"),
        Err(source) => {
            return Err(DistributedRuntimeError::Transport {
                context: context.to_owned(),
                source,
            });
        }
    }
    stream
        .read_exact(&mut length_bytes[1..])
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

    Ok(Some(serde_json::from_slice(&bytes)?))
}

fn read_message<T: DeserializeOwned>(
    stream: &mut TcpStream,
    context: &str,
) -> Result<T, DistributedRuntimeError> {
    try_read_message(stream, context)?.ok_or_else(|| DistributedRuntimeError::Transport {
        context: context.to_owned(),
        source: io::Error::new(ErrorKind::UnexpectedEof, "connection closed"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs_runtime::checkpoint::StableTopologyPointers;
    use crate::ecs_runtime::components::LocalWeights;

    const EPSILON: f32 = 1.0e-6;

    #[test]
    fn partitions_nodes_without_empty_shards() {
        let assignments = partition_node_ranges(
            &[
                "worker-a:7000".to_owned(),
                "worker-b:7000".to_owned(),
                "worker-c:7000".to_owned(),
            ],
            5,
        )
        .expect("partition node ranges");

        assert_eq!(assignments.len(), 3);
        assert_eq!(assignments[0].1, 0..2);
        assert_eq!(assignments[1].1, 2..4);
        assert_eq!(assignments[2].1, 4..5);
    }

    #[test]
    fn worker_shard_state_updates_forward_activation_and_weights() {
        let mut shard = WorkerShardState::initialize(vec![GraphNodeCheckpoint {
            index: 0,
            activation: 0.0,
            is_input: false,
            local_weights: LocalWeights::new([0.5, -0.25, 0.1]),
            topology: StableTopologyPointers::new([1, 2, 3]),
        }])
        .expect("initialize shard");
        let neighbor_activations = [1.0, -0.5, 0.25];

        let activation = shard
            .forward_node(ForwardNodeRequest {
                node_index: 0,
                neighbor_activations,
                activation_kind: ActivationKind::Relu,
            })
            .expect("forward node")
            .activation;
        assert!((activation - 0.65).abs() < EPSILON);

        let response = shard
            .update_weights_batch(UpdateWeightsBatchRequest {
                updates: vec![WeightUpdateNodeRequest {
                    node_index: 0,
                    neighbor_activations,
                    phase: SimulationPhase::Positive,
                    learning_rate: 0.1,
                }],
            })
            .expect("update weights");
        let weights = response.updates[0].neighbor_weights;

        assert!((weights[0] - 0.612_775).abs() < EPSILON);
        assert!((weights[1] + 0.306_387_5).abs() < EPSILON);
        assert!((weights[2] - 0.128_193_75).abs() < EPSILON);
    }

    #[test]
    fn worker_protocol_round_trips_over_tcp() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral listener");
        let addr = listener.local_addr().expect("listener addr");
        let handle = thread::spawn(move || {
            let (stream, peer) = listener.accept().expect("accept connection");
            handle_worker_connection(stream, &peer.to_string()).expect("serve connection");
        });

        let mut client = PartitionedWorkerClient::connect(
            addr.to_string(),
            0..1,
            vec![GraphNodeCheckpoint {
                index: 0,
                activation: 0.0,
                is_input: true,
                local_weights: LocalWeights::new([1.0, 0.0, 0.0]),
                topology: StableTopologyPointers::new([0, 0, 0]),
            }],
        )
        .expect("connect worker client");

        client
            .set_node_activations(vec![NodeActivationPatch {
                node_index: 0,
                activation: 2.0,
            }])
            .expect("set activation");
        let activation = client
            .forward_node(0, [1.5, 0.0, 0.0], ActivationKind::Relu)
            .expect("forward node");

        assert_eq!(activation, 1.5);

        drop(client);
        handle.join().expect("join worker thread");
    }
}
