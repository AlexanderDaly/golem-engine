use std::env;
use std::path::PathBuf;
use std::process;

use hecs::{Entity, NoSuchEntity, World};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use thiserror::Error;

use golem_engine::core_math::ramanujan_gen::{
    generate_verified_regular_graph, GraphGenerationError, VerifiedRamanujanGraph,
};
use golem_engine::data::mnist_loader::{MnistDataset, MnistLoadError};
use golem_engine::ecs_runtime::components::{InputNode, LocalWeights, NodeState, TopologyPointers};
use golem_engine::ecs_runtime::ingestion_system::{
    inject_data_system, ContrastiveDataStream, IngestionError, SimulationPhase,
};
use golem_engine::ecs_runtime::systems::{
    update_local_weights_forward_forward, update_nodes_forward_forward, ActivationKind,
    NodeUpdateError,
};
use golem_engine::{NODE_COUNT, REGULAR_DEGREE};

const DEFAULT_EPOCHS: usize = 1;
const DEFAULT_LEARNING_RATE: f32 = 1.0e-3;
const DEFAULT_GRAPH_SEARCH_LIMIT: u64 = 50_000;
const DEFAULT_WEIGHT_SEED: u64 = 0;
const DEFAULT_WEIGHT_INIT_SCALE: f32 = 0.05;

fn main() {
    match run() {
        Ok(()) => {}
        Err(AppError::HelpRequested { program }) => {
            print_usage(&program);
        }
        Err(error) => {
            eprintln!("error: {error}");
            process::exit(1);
        }
    }
}

fn run() -> Result<(), AppError> {
    let config = TrainingConfig::from_args(env::args())?;

    println!(
        "loading MNIST dataset from {} and {}",
        config.train_images.display(),
        config.train_labels.display()
    );
    let dataset = MnistDataset::load(&config.train_images, &config.train_labels)?;
    println!("loaded {} labeled samples", dataset.len());

    println!(
        "searching deterministic seeds for a verified {}-node, {}-regular Ramanujan graph (limit: {})",
        NODE_COUNT, REGULAR_DEGREE, config.graph_search_limit
    );
    let graph =
        generate_verified_regular_graph(NODE_COUNT, REGULAR_DEGREE, config.graph_search_limit)?;
    println!(
        "using graph from seed {} with second-largest |eigenvalue| {:.6} (bound {:.6})",
        graph.certificate.search_seed,
        graph.certificate.second_largest_absolute_eigenvalue,
        graph.certificate.ramanujan_bound
    );

    let mut world = build_training_world(&graph, config.weight_seed, config.weight_init_scale)?;
    let mut stream = ContrastiveDataStream::new(dataset)?;
    let ticks_per_epoch =
        stream
            .dataset()
            .len()
            .checked_mul(2)
            .ok_or(AppError::TickCountOverflow {
                dataset_len: stream.dataset().len(),
                epochs: config.epochs,
            })?;
    let total_ticks =
        ticks_per_epoch
            .checked_mul(config.epochs)
            .ok_or(AppError::TickCountOverflow {
                dataset_len: stream.dataset().len(),
                epochs: config.epochs,
            })?;

    println!(
        "starting training: epochs={} ticks_per_epoch={} total_ticks={} activation={} learning_rate={} weight_seed={} weight_init_scale={}",
        config.epochs,
        ticks_per_epoch,
        total_ticks,
        activation_name(config.activation_kind),
        config.learning_rate,
        config.weight_seed,
        config.weight_init_scale
    );

    let mut phase = SimulationPhase::Positive;
    for epoch in 0..config.epochs {
        for _tick in 0..ticks_per_epoch {
            let sample_phase = phase;
            inject_data_system(&mut world, &mut stream, &mut phase)?;
            update_nodes_forward_forward(&mut world, config.activation_kind)?;
            update_local_weights_forward_forward(&mut world, sample_phase, config.learning_rate)?;
        }

        let summary = summarize_world(&world);
        println!(
            "epoch {}/{} complete | mean_abs_activation={:.6} mean_squared_activation={:.6} mean_abs_weight={:.6}",
            epoch + 1,
            config.epochs,
            summary.mean_abs_activation,
            summary.mean_squared_activation,
            summary.mean_abs_weight
        );
    }

    println!("training complete");

    Ok(())
}

#[derive(Debug, Clone)]
struct TrainingConfig {
    train_images: PathBuf,
    train_labels: PathBuf,
    epochs: usize,
    learning_rate: f32,
    activation_kind: ActivationKind,
    graph_search_limit: u64,
    weight_seed: u64,
    weight_init_scale: f32,
}

impl TrainingConfig {
    fn from_args<I>(args: I) -> Result<Self, AppError>
    where
        I: IntoIterator<Item = String>,
    {
        let mut args = args.into_iter();
        let program = args.next().unwrap_or_else(|| "golem-engine".to_owned());

        let mut train_images = None;
        let mut train_labels = None;
        let mut epochs = DEFAULT_EPOCHS;
        let mut learning_rate = DEFAULT_LEARNING_RATE;
        let mut activation_kind = ActivationKind::Tanh;
        let mut graph_search_limit = DEFAULT_GRAPH_SEARCH_LIMIT;
        let mut weight_seed = DEFAULT_WEIGHT_SEED;
        let mut weight_init_scale = DEFAULT_WEIGHT_INIT_SCALE;

        let remaining: Vec<String> = args.collect();
        if remaining.is_empty() {
            return Err(AppError::HelpRequested { program });
        }

        let mut index = 0usize;
        while index < remaining.len() {
            match remaining[index].as_str() {
                "--help" | "-h" => return Err(AppError::HelpRequested { program }),
                "--train-images" => {
                    train_images = Some(PathBuf::from(argument_value(&remaining, &mut index)?));
                }
                "--train-labels" => {
                    train_labels = Some(PathBuf::from(argument_value(&remaining, &mut index)?));
                }
                "--epochs" => {
                    epochs = parse_usize(argument_value(&remaining, &mut index)?, "--epochs")?;
                    if epochs == 0 {
                        return Err(AppError::InvalidArgument {
                            flag: "--epochs",
                            value: "0".to_owned(),
                            reason: "must be greater than zero",
                        });
                    }
                }
                "--learning-rate" => {
                    learning_rate =
                        parse_f32(argument_value(&remaining, &mut index)?, "--learning-rate")?;
                    if learning_rate <= 0.0 {
                        return Err(AppError::InvalidArgument {
                            flag: "--learning-rate",
                            value: learning_rate.to_string(),
                            reason: "must be greater than zero",
                        });
                    }
                }
                "--activation" => {
                    activation_kind =
                        parse_activation_kind(argument_value(&remaining, &mut index)?)?;
                }
                "--graph-search-limit" => {
                    graph_search_limit = parse_u64(
                        argument_value(&remaining, &mut index)?,
                        "--graph-search-limit",
                    )?;
                    if graph_search_limit == 0 {
                        return Err(AppError::InvalidArgument {
                            flag: "--graph-search-limit",
                            value: "0".to_owned(),
                            reason: "must be greater than zero",
                        });
                    }
                }
                "--weight-seed" => {
                    weight_seed =
                        parse_u64(argument_value(&remaining, &mut index)?, "--weight-seed")?;
                }
                "--weight-init-scale" => {
                    weight_init_scale = parse_f32(
                        argument_value(&remaining, &mut index)?,
                        "--weight-init-scale",
                    )?;
                    if weight_init_scale < 0.0 {
                        return Err(AppError::InvalidArgument {
                            flag: "--weight-init-scale",
                            value: weight_init_scale.to_string(),
                            reason: "must be non-negative",
                        });
                    }
                }
                flag => {
                    return Err(AppError::UnknownFlag {
                        flag: flag.to_owned(),
                    });
                }
            }

            index += 1;
        }

        Ok(Self {
            train_images: train_images.ok_or(AppError::MissingRequiredArgument {
                flag: "--train-images",
            })?,
            train_labels: train_labels.ok_or(AppError::MissingRequiredArgument {
                flag: "--train-labels",
            })?,
            epochs,
            learning_rate,
            activation_kind,
            graph_search_limit,
            weight_seed,
            weight_init_scale,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct TrainingSummary {
    mean_abs_activation: f32,
    mean_squared_activation: f32,
    mean_abs_weight: f32,
}

#[derive(Debug, Error)]
enum AppError {
    #[error("")]
    HelpRequested { program: String },
    #[error("missing required argument {flag}")]
    MissingRequiredArgument { flag: &'static str },
    #[error("missing value for {flag}")]
    MissingValue { flag: String },
    #[error("unknown flag {flag}")]
    UnknownFlag { flag: String },
    #[error("invalid value {value:?} for {flag}: {reason}")]
    InvalidArgument {
        flag: &'static str,
        value: String,
        reason: &'static str,
    },
    #[error("failed to parse {flag} value {value:?} as an integer")]
    InvalidInteger { flag: &'static str, value: String },
    #[error("failed to parse {flag} value {value:?} as a float")]
    InvalidFloat { flag: &'static str, value: String },
    #[error("unsupported activation kind {value:?}; expected one of: tanh, relu, softsign")]
    InvalidActivation { value: String },
    #[error("failed to load MNIST dataset: {0}")]
    MnistLoad(#[from] MnistLoadError),
    #[error("failed to build contrastive stream: {0}")]
    Ingestion(#[from] IngestionError),
    #[error("failed to generate a verified graph: {0}")]
    GraphGeneration(#[from] GraphGenerationError),
    #[error("training step failed: {0}")]
    NodeUpdate(#[from] NodeUpdateError),
    #[error("graph edge ({u}, {v}) references an out-of-bounds node for node_count {node_count}")]
    InvalidGraphEdge {
        u: usize,
        v: usize,
        node_count: usize,
    },
    #[error("graph node {node} has {actual} neighbors, expected {expected}")]
    InvalidNeighborCount {
        node: usize,
        actual: usize,
        expected: usize,
    },
    #[error("failed to attach components to entity for graph node {node}: {source}")]
    EntityInsertion {
        node: usize,
        #[source]
        source: NoSuchEntity,
    },
    #[error("training tick count overflow for dataset_len={dataset_len} epochs={epochs}")]
    TickCountOverflow { dataset_len: usize, epochs: usize },
}

fn argument_value<'a>(args: &'a [String], index: &mut usize) -> Result<&'a str, AppError> {
    let flag = args[*index].clone();
    *index += 1;
    args.get(*index)
        .map(String::as_str)
        .ok_or(AppError::MissingValue { flag })
}

fn parse_usize(value: &str, flag: &'static str) -> Result<usize, AppError> {
    value
        .parse::<usize>()
        .map_err(|_| AppError::InvalidInteger {
            flag,
            value: value.to_owned(),
        })
}

fn parse_u64(value: &str, flag: &'static str) -> Result<u64, AppError> {
    value.parse::<u64>().map_err(|_| AppError::InvalidInteger {
        flag,
        value: value.to_owned(),
    })
}

fn parse_f32(value: &str, flag: &'static str) -> Result<f32, AppError> {
    value.parse::<f32>().map_err(|_| AppError::InvalidFloat {
        flag,
        value: value.to_owned(),
    })
}

fn parse_activation_kind(value: &str) -> Result<ActivationKind, AppError> {
    match value {
        "tanh" => Ok(ActivationKind::Tanh),
        "relu" => Ok(ActivationKind::Relu),
        "softsign" => Ok(ActivationKind::SoftSign),
        _ => Err(AppError::InvalidActivation {
            value: value.to_owned(),
        }),
    }
}

fn activation_name(kind: ActivationKind) -> &'static str {
    match kind {
        ActivationKind::Tanh => "tanh",
        ActivationKind::Relu => "relu",
        ActivationKind::SoftSign => "softsign",
    }
}

fn build_training_world(
    graph: &VerifiedRamanujanGraph,
    weight_seed: u64,
    weight_init_scale: f32,
) -> Result<World, AppError> {
    let node_count = graph.certificate.node_count;
    let adjacency = build_adjacency_lists(node_count, &graph.edges)?;
    let mut world = World::new();
    let entities: Vec<Entity> = (0..node_count)
        .map(|_| world.spawn((InputNode, NodeState::new(0.0))))
        .collect();
    let mut rng = ChaCha8Rng::seed_from_u64(weight_seed);

    for (node_index, entity) in entities.iter().copied().enumerate() {
        let neighbors: [usize; REGULAR_DEGREE] = adjacency[node_index]
            .as_slice()
            .try_into()
            .map_err(|_| AppError::InvalidNeighborCount {
                node: node_index,
                actual: adjacency[node_index].len(),
                expected: REGULAR_DEGREE,
            })?;
        let topology =
            TopologyPointers::new(neighbors.map(|neighbor_index| entities[neighbor_index]));
        let weights = LocalWeights::new(std::array::from_fn(|_| {
            rng.gen_range(-weight_init_scale..=weight_init_scale)
        }));

        world
            .insert(entity, (weights, topology))
            .map_err(|source| AppError::EntityInsertion {
                node: node_index,
                source,
            })?;
    }

    Ok(world)
}

fn build_adjacency_lists(
    node_count: usize,
    edges: &[(usize, usize)],
) -> Result<Vec<Vec<usize>>, AppError> {
    let mut adjacency = vec![Vec::with_capacity(REGULAR_DEGREE); node_count];

    for &(u, v) in edges {
        if u >= node_count || v >= node_count {
            return Err(AppError::InvalidGraphEdge { u, v, node_count });
        }

        adjacency[u].push(v);
        adjacency[v].push(u);
    }

    for (node, neighbors) in adjacency.iter_mut().enumerate() {
        neighbors.sort_unstable();
        if neighbors.len() != REGULAR_DEGREE {
            return Err(AppError::InvalidNeighborCount {
                node,
                actual: neighbors.len(),
                expected: REGULAR_DEGREE,
            });
        }
    }

    Ok(adjacency)
}

fn summarize_world(world: &World) -> TrainingSummary {
    let mut activation_count = 0usize;
    let mut abs_activation_sum = 0.0f32;
    let mut squared_activation_sum = 0.0f32;

    for (_entity, state) in world.query::<&NodeState>().iter() {
        activation_count += 1;
        abs_activation_sum += state.activation.abs();
        squared_activation_sum += state.activation * state.activation;
    }

    let mut weight_count = 0usize;
    let mut abs_weight_sum = 0.0f32;
    for (_entity, weights) in world.query::<&LocalWeights>().iter() {
        for weight in weights.neighbor_weights {
            weight_count += 1;
            abs_weight_sum += weight.abs();
        }
    }

    TrainingSummary {
        mean_abs_activation: abs_activation_sum / activation_count as f32,
        mean_squared_activation: squared_activation_sum / activation_count as f32,
        mean_abs_weight: abs_weight_sum / weight_count as f32,
    }
}

fn print_usage(program: &str) {
    println!(
        "Usage:
  {program} --train-images <PATH> --train-labels <PATH> [options]

Required:
  --train-images <PATH>      Path to MNIST image IDX data
  --train-labels <PATH>      Path to MNIST label IDX data

Options:
  --epochs <N>               Number of training epochs to run (default: {DEFAULT_EPOCHS})
  --learning-rate <F32>      Forward-Forward local learning rate (default: {DEFAULT_LEARNING_RATE})
  --activation <NAME>        Node activation: tanh | relu | softsign (default: tanh)
  --graph-search-limit <N>   Deterministic seed search budget for Ramanujan graph generation (default: {DEFAULT_GRAPH_SEARCH_LIMIT})
  --weight-seed <N>          RNG seed for local weight initialization (default: {DEFAULT_WEIGHT_SEED})
  --weight-init-scale <F32>  Uniform half-range for initial weights (default: {DEFAULT_WEIGHT_INIT_SCALE})
  --help, -h                 Show this help text

Training schedule per tick:
  inject_data_system -> update_nodes_forward_forward -> update_local_weights_forward_forward

Epoch semantics:
  one epoch runs 2 * dataset_len ticks so both the positive and negative cursors traverse the full dataset once."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_required_training_paths_and_defaults() {
        let config = TrainingConfig::from_args([
            "golem-engine".to_owned(),
            "--train-images".to_owned(),
            "train-images-idx3-ubyte".to_owned(),
            "--train-labels".to_owned(),
            "train-labels-idx1-ubyte".to_owned(),
        ])
        .expect("parse config");

        assert_eq!(
            config.train_images,
            PathBuf::from("train-images-idx3-ubyte")
        );
        assert_eq!(
            config.train_labels,
            PathBuf::from("train-labels-idx1-ubyte")
        );
        assert_eq!(config.epochs, DEFAULT_EPOCHS);
        assert_eq!(config.learning_rate, DEFAULT_LEARNING_RATE);
        assert!(matches!(config.activation_kind, ActivationKind::Tanh));
    }

    #[test]
    fn rejects_missing_required_arguments() {
        let error = TrainingConfig::from_args([
            "golem-engine".to_owned(),
            "--train-images".to_owned(),
            "train-images-idx3-ubyte".to_owned(),
        ])
        .expect_err("missing labels should fail");

        assert!(matches!(
            error,
            AppError::MissingRequiredArgument {
                flag: "--train-labels"
            }
        ));
    }
}
