pub mod cli;
pub mod eval;
pub mod experiment;
pub mod world;

use std::path::PathBuf;
use std::time::Instant;

use thiserror::Error;

use crate::core_math::ramanujan_gen::{
    generate_verified_regular_graph, recommended_seed_search_limit, GraphGenerationError,
};
use crate::data::mnist_loader::{MnistDataset, MnistLoadError};
use crate::ecs_runtime::checkpoint::{load_checkpoint_json, save_checkpoint_json, CheckpointError};
use crate::ecs_runtime::ingestion_system::{
    inject_data_system, ContrastiveDataStream, IngestionError, SimulationPhase,
};
use crate::ecs_runtime::systems::{
    update_local_weights_forward_forward, update_nodes_forward_forward, NodeUpdateError,
};
use crate::REGULAR_DEGREE;

pub use cli::{print_usage, TrainingConfig, TrainingConfigError};
pub use eval::{evaluate_forward_forward, EvaluationError, EvaluationSummary};
pub use experiment::{
    CheckpointArtifact, CheckpointSettings, EffectiveRunSettings, ExperimentError, ExperimentRun,
    MetricsRecord, RunManifest,
};
pub use world::{
    build_training_world, count_input_nodes, count_nodes, summarize_world,
    training_input_node_count, WorldBuildError, WorldSummary,
};

#[derive(Debug, Clone)]
pub struct TrainingRunOutput {
    pub run_dir: std::path::PathBuf,
    pub manifest_path: std::path::PathBuf,
    pub metrics_path: std::path::PathBuf,
    pub latest_checkpoint_path: Option<std::path::PathBuf>,
}

pub fn run_training(config: TrainingConfig) -> Result<TrainingRunOutput, TrainingRunError> {
    println!(
        "loading MNIST training dataset from {} and {}",
        config.train.images.display(),
        config.train.labels.display()
    );
    let train_dataset = MnistDataset::load(&config.train.images, &config.train.labels)?;
    println!("loaded {} training samples", train_dataset.len());

    let test_dataset = if let Some(test) = &config.test {
        println!(
            "loading MNIST evaluation dataset from {} and {}",
            test.images.display(),
            test.labels.display()
        );
        let dataset = MnistDataset::load(&test.images, &test.labels)?;
        println!("loaded {} evaluation samples", dataset.len());
        Some(dataset)
    } else {
        None
    };

    let mut effective_graph_search_limit = None;
    let (mut world, mut phase) = if let Some(path) = &config.load_checkpoint {
        println!("loading checkpoint from {}", path.display());
        println!(
            "resuming from serialized world state; graph generation and weight initialization flags are ignored"
        );
        load_checkpoint_json(path)?
    } else {
        let graph_search_limit = config
            .graph_search_limit
            .unwrap_or_else(|| recommended_seed_search_limit(config.graph_node_count));
        let input_node_count = training_input_node_count(config.graph_node_count);
        effective_graph_search_limit = Some(graph_search_limit);

        println!(
            "searching deterministic seeds for a verified {}-node, {}-regular Ramanujan graph (limit: {})",
            config.graph_node_count,
            REGULAR_DEGREE,
            graph_search_limit
        );
        let graph = generate_verified_regular_graph(
            config.graph_node_count,
            REGULAR_DEGREE,
            graph_search_limit,
        )?;
        println!(
            "using graph from seed {} with second-largest |eigenvalue| {:.6} (bound {:.6}); input_nodes={}",
            graph.certificate.search_seed,
            graph.certificate.second_largest_absolute_eigenvalue,
            graph.certificate.ramanujan_bound,
            input_node_count
        );

        (
            build_training_world(
                &graph,
                config.weight_seed,
                config.weight_init_scale,
                input_node_count,
            )?,
            SimulationPhase::Positive,
        )
    };

    let world_node_count = count_nodes(&world);
    let input_node_count = count_input_nodes(&world);
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

    println!("run directory: {}", experiment.root().display());
    println!(
        "starting training: start_epoch={} end_epoch={} epochs_this_invocation={} ticks_per_epoch={} total_ticks={} activation={} learning_rate={} graph_node_count={} input_node_count={} weight_seed={} weight_init_scale={} eval_every={} checkpoint_every={}",
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
    );

    for epoch_offset in 0..config.epochs {
        let epoch = experiment.starting_epoch() + epoch_offset + 1;

        for _tick in 0..ticks_per_epoch {
            let sample_phase = phase;
            inject_data_system(&mut world, &mut stream, &mut phase)?;
            update_nodes_forward_forward(&mut world, config.activation_kind)?;
            update_local_weights_forward_forward(&mut world, sample_phase, config.learning_rate)?;
        }

        let summary = summarize_world(&world);
        let evaluation = if epoch % config.eval_every == 0 {
            Some(evaluate_forward_forward(
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
                "epoch {epoch}/{final_epoch} complete | mean_abs_activation={:.6} mean_squared_activation={:.6} mean_abs_weight={:.6} mean_positive_goodness={:.6} mean_negative_goodness={:.6} goodness_separation={:.6}",
                summary.mean_abs_activation,
                summary.mean_squared_activation,
                summary.mean_abs_weight,
                evaluation.mean_positive_goodness,
                evaluation.mean_negative_goodness,
                evaluation.goodness_separation,
            );
        } else {
            println!(
                "epoch {epoch}/{final_epoch} complete | mean_abs_activation={:.6} mean_squared_activation={:.6} mean_abs_weight={:.6} eval=skipped",
                summary.mean_abs_activation,
                summary.mean_squared_activation,
                summary.mean_abs_weight,
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
        latest_checkpoint_path: experiment.latest_checkpoint_path().map(PathBuf::from),
    })
}

#[derive(Debug, Error)]
pub enum TrainingRunError {
    #[error("failed to load MNIST dataset: {0}")]
    MnistLoad(#[from] MnistLoadError),
    #[error("failed to build contrastive stream: {0}")]
    Ingestion(#[from] IngestionError),
    #[error("failed to generate a verified graph: {0}")]
    GraphGeneration(#[from] GraphGenerationError),
    #[error("checkpoint operation failed: {0}")]
    Checkpoint(#[from] CheckpointError),
    #[error("training step failed: {0}")]
    NodeUpdate(#[from] NodeUpdateError),
    #[error("failed to build the training world: {0}")]
    WorldBuild(#[from] WorldBuildError),
    #[error("experiment management failed: {0}")]
    Experiment(#[from] ExperimentError),
    #[error("forward-forward evaluation failed: {0}")]
    Evaluation(#[from] EvaluationError),
    #[error("training tick count overflow for dataset_len={dataset_len} epochs={epochs}")]
    TickCountOverflow { dataset_len: usize, epochs: usize },
}
