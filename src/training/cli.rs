use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ecs_runtime::systems::{ActivationKind, ParseActivationKindError};
use crate::CONDITIONED_INPUT_NODE_COUNT;

pub const DEFAULT_EPOCHS: usize = 1;
pub const DEFAULT_LEARNING_RATE: f32 = 1.0e-3;
pub const DEFAULT_GRAPH_NODE_COUNT: usize = CONDITIONED_INPUT_NODE_COUNT;
pub const DEFAULT_WEIGHT_SEED: u64 = 0;
pub const DEFAULT_WEIGHT_INIT_SCALE: f32 = 0.05;
pub const DEFAULT_EVAL_EVERY: usize = 1;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DatasetPaths {
    pub images: PathBuf,
    pub labels: PathBuf,
}

#[derive(Debug, Clone)]
pub struct WorkerServerConfig {
    pub listen_addr: String,
}

#[derive(Debug, Clone)]
pub enum CliCommand {
    Train(TrainingConfig),
    Worker(WorkerServerConfig),
}

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub train: DatasetPaths,
    pub test: Option<DatasetPaths>,
    pub epochs: usize,
    pub learning_rate: f32,
    pub activation_kind: ActivationKind,
    pub graph_node_count: usize,
    pub graph_search_limit: Option<u64>,
    pub weight_seed: u64,
    pub weight_init_scale: f32,
    pub run_dir: Option<PathBuf>,
    pub checkpoint_every: Option<usize>,
    pub eval_every: usize,
    pub load_checkpoint: Option<PathBuf>,
    pub save_checkpoint: Option<PathBuf>,
    pub distributed_workers: Vec<String>,
}

impl CliCommand {
    pub fn from_args<I>(args: I) -> Result<Self, TrainingConfigError>
    where
        I: IntoIterator<Item = String>,
    {
        let mut args = args.into_iter();
        let program = args.next().unwrap_or_else(|| "golem-engine".to_owned());

        let mut train_images = None;
        let mut train_labels = None;
        let mut test_images = None;
        let mut test_labels = None;
        let mut epochs = DEFAULT_EPOCHS;
        let mut learning_rate = DEFAULT_LEARNING_RATE;
        let mut activation_kind = ActivationKind::Tanh;
        let mut graph_node_count = DEFAULT_GRAPH_NODE_COUNT;
        let mut graph_search_limit = None;
        let mut weight_seed = DEFAULT_WEIGHT_SEED;
        let mut weight_init_scale = DEFAULT_WEIGHT_INIT_SCALE;
        let mut run_dir = None;
        let mut checkpoint_every = None;
        let mut eval_every = DEFAULT_EVAL_EVERY;
        let mut load_checkpoint = None;
        let mut save_checkpoint = None;
        let mut distributed_workers = Vec::new();
        let mut worker_listen = None;

        let remaining: Vec<String> = args.collect();
        if remaining.is_empty() {
            return Err(TrainingConfigError::HelpRequested { program });
        }

        let mut index = 0usize;
        while index < remaining.len() {
            match remaining[index].as_str() {
                "--help" | "-h" => return Err(TrainingConfigError::HelpRequested { program }),
                "--train-images" => {
                    train_images = Some(PathBuf::from(argument_value(&remaining, &mut index)?));
                }
                "--train-labels" => {
                    train_labels = Some(PathBuf::from(argument_value(&remaining, &mut index)?));
                }
                "--test-images" => {
                    test_images = Some(PathBuf::from(argument_value(&remaining, &mut index)?));
                }
                "--test-labels" => {
                    test_labels = Some(PathBuf::from(argument_value(&remaining, &mut index)?));
                }
                "--epochs" => {
                    epochs = parse_usize(argument_value(&remaining, &mut index)?, "--epochs")?;
                    if epochs == 0 {
                        return Err(TrainingConfigError::InvalidArgument {
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
                        return Err(TrainingConfigError::InvalidArgument {
                            flag: "--learning-rate",
                            value: learning_rate.to_string(),
                            reason: "must be greater than zero",
                        });
                    }
                }
                "--activation" => {
                    activation_kind = argument_value(&remaining, &mut index)?.parse().map_err(
                        |source: ParseActivationKindError| TrainingConfigError::InvalidActivation {
                            value: source.value,
                        },
                    )?;
                }
                "--graph-node-count" => {
                    graph_node_count = parse_usize(
                        argument_value(&remaining, &mut index)?,
                        "--graph-node-count",
                    )?;
                    if graph_node_count == 0 {
                        return Err(TrainingConfigError::InvalidArgument {
                            flag: "--graph-node-count",
                            value: "0".to_owned(),
                            reason: "must be greater than zero",
                        });
                    }
                }
                "--graph-search-limit" => {
                    graph_search_limit = Some(parse_u64(
                        argument_value(&remaining, &mut index)?,
                        "--graph-search-limit",
                    )?);
                    if graph_search_limit == Some(0) {
                        return Err(TrainingConfigError::InvalidArgument {
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
                        return Err(TrainingConfigError::InvalidArgument {
                            flag: "--weight-init-scale",
                            value: weight_init_scale.to_string(),
                            reason: "must be non-negative",
                        });
                    }
                }
                "--run-dir" => {
                    run_dir = Some(PathBuf::from(argument_value(&remaining, &mut index)?));
                }
                "--checkpoint-every" => {
                    checkpoint_every = Some(parse_usize(
                        argument_value(&remaining, &mut index)?,
                        "--checkpoint-every",
                    )?);
                    if checkpoint_every == Some(0) {
                        return Err(TrainingConfigError::InvalidArgument {
                            flag: "--checkpoint-every",
                            value: "0".to_owned(),
                            reason: "must be greater than zero",
                        });
                    }
                }
                "--eval-every" => {
                    eval_every =
                        parse_usize(argument_value(&remaining, &mut index)?, "--eval-every")?;
                    if eval_every == 0 {
                        return Err(TrainingConfigError::InvalidArgument {
                            flag: "--eval-every",
                            value: "0".to_owned(),
                            reason: "must be greater than zero",
                        });
                    }
                }
                "--load-checkpoint" => {
                    load_checkpoint = Some(PathBuf::from(argument_value(&remaining, &mut index)?));
                }
                "--save-checkpoint" => {
                    save_checkpoint = Some(PathBuf::from(argument_value(&remaining, &mut index)?));
                }
                "--distributed-worker" => {
                    distributed_workers.push(argument_value(&remaining, &mut index)?.to_owned());
                }
                "--worker-listen" => {
                    worker_listen = Some(argument_value(&remaining, &mut index)?.to_owned());
                }
                flag => {
                    return Err(TrainingConfigError::UnknownFlag {
                        flag: flag.to_owned(),
                    });
                }
            }

            index += 1;
        }

        if let Some(listen_addr) = worker_listen {
            if !distributed_workers.is_empty() {
                return Err(TrainingConfigError::InvalidFlagCombination {
                    flag: "--worker-listen",
                    conflicting_flag: "--distributed-worker",
                });
            }

            return Ok(Self::Worker(WorkerServerConfig { listen_addr }));
        }

        let train = DatasetPaths {
            images: train_images.ok_or(TrainingConfigError::MissingRequiredArgument {
                flag: "--train-images",
            })?,
            labels: train_labels.ok_or(TrainingConfigError::MissingRequiredArgument {
                flag: "--train-labels",
            })?,
        };
        let test = match (test_images, test_labels) {
            (Some(images), Some(labels)) => Some(DatasetPaths { images, labels }),
            (Some(_), None) => {
                return Err(TrainingConfigError::MissingRequiredCompanionArgument {
                    flag: "--test-images",
                    companion_flag: "--test-labels",
                });
            }
            (None, Some(_)) => {
                return Err(TrainingConfigError::MissingRequiredCompanionArgument {
                    flag: "--test-labels",
                    companion_flag: "--test-images",
                });
            }
            (None, None) => None,
        };

        Ok(Self::Train(TrainingConfig {
            train,
            test,
            epochs,
            learning_rate,
            activation_kind,
            graph_node_count,
            graph_search_limit,
            weight_seed,
            weight_init_scale,
            run_dir,
            checkpoint_every,
            eval_every,
            load_checkpoint,
            save_checkpoint,
            distributed_workers,
        }))
    }
}

impl TrainingConfig {
    pub fn from_args<I>(args: I) -> Result<Self, TrainingConfigError>
    where
        I: IntoIterator<Item = String>,
    {
        match CliCommand::from_args(args)? {
            CliCommand::Train(config) => Ok(config),
            CliCommand::Worker(config) => Err(TrainingConfigError::WorkerModeRequested {
                listen_addr: config.listen_addr,
            }),
        }
    }
}

#[derive(Debug, Error)]
pub enum TrainingConfigError {
    #[error("")]
    HelpRequested { program: String },
    #[error("missing required argument {flag}")]
    MissingRequiredArgument { flag: &'static str },
    #[error("{flag} requires {companion_flag} to be provided as well")]
    MissingRequiredCompanionArgument {
        flag: &'static str,
        companion_flag: &'static str,
    },
    #[error("missing value for {flag}")]
    MissingValue { flag: String },
    #[error("unknown flag {flag}")]
    UnknownFlag { flag: String },
    #[error("{flag} cannot be combined with {conflicting_flag}")]
    InvalidFlagCombination {
        flag: &'static str,
        conflicting_flag: &'static str,
    },
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
    #[error(
        "worker-server mode requested on {listen_addr}; parse with CliCommand::from_args instead"
    )]
    WorkerModeRequested { listen_addr: String },
}

pub fn print_usage(program: &str) {
    println!(
        "Usage:
  {program} --train-images <PATH> --train-labels <PATH> [options]
  {program} --worker-listen <HOST:PORT>

Required:
  --train-images <PATH>      Path to MNIST training image IDX data
  --train-labels <PATH>      Path to MNIST training label IDX data

Optional evaluation dataset:
  --test-images <PATH>       Path to MNIST evaluation image IDX data
  --test-labels <PATH>       Path to MNIST evaluation label IDX data

Experiment management:
  --run-dir <PATH>           Root directory for manifest, metrics, and checkpoints
  --checkpoint-every <N>     Save run-directory checkpoints every N epochs and on the final epoch
  --eval-every <N>           Run the no-learning FF evaluation every N epochs (default: {DEFAULT_EVAL_EVERY})
  --load-checkpoint <PATH>   Restore a previously saved graph state instead of generating a new graph
  --save-checkpoint <PATH>   Persist the final graph state after training
  --distributed-worker <A>   Remote worker address (repeatable) for distributed federated averaging

Training options:
  --epochs <N>               Number of epochs to run for this invocation (default: {DEFAULT_EPOCHS})
  --learning-rate <F32>      Forward-Forward local learning rate (default: {DEFAULT_LEARNING_RATE})
  --activation <NAME>        Node activation: tanh | relu | softsign (default: tanh)
  --graph-node-count <N>     Number of graph nodes to generate (default: {DEFAULT_GRAPH_NODE_COUNT})
  --graph-search-limit <N>   Deterministic seed search budget for Ramanujan graph generation
  --weight-seed <N>          RNG seed for local weight initialization (default: {DEFAULT_WEIGHT_SEED})
  --weight-init-scale <F32>  Uniform half-range for initial weights (default: {DEFAULT_WEIGHT_INIT_SCALE})
  --help, -h                 Show this help text

Worker mode:
  --worker-listen <HOST:PORT>
                            Run a distributed worker server instead of local training

Run-directory behavior:
  if --run-dir is omitted, the binary creates runs/<unix-timestamp>-<short-id>/.
  when resuming from runs/.../checkpoints/epoch-*.json, the same run directory is reused when that would not overwrite newer metrics.

Training schedule per tick:
    inject_data_system -> update_nodes_forward_forward -> update_local_weights_forward_forward

Evaluation schedule:
  reset activations -> inject conditioned sample -> update_nodes_forward_forward -> score all candidate labels

Epoch semantics:
  one epoch runs 2 * dataset_len ticks so both the positive and negative cursors traverse the full dataset once.

Input semantics:
  fresh runs reserve the first 794 nodes as conditioned inputs: slots 0..783 are MNIST pixels and slots 784..793 are one-hot label inputs.
  label-conditioned training requires graph_node_count >= 794.

Distributed execution:
  distributed mode snapshots the world once per epoch, sends identical checkpoints to remote workers, and averages the worker-updated worlds back on the coordinator.
  every worker must be able to read the same dataset paths passed to --train-images/--train-labels and --test-images/--test-labels."
    );
}

fn argument_value<'a>(
    args: &'a [String],
    index: &mut usize,
) -> Result<&'a str, TrainingConfigError> {
    let flag = args[*index].clone();
    *index += 1;
    args.get(*index)
        .map(String::as_str)
        .ok_or(TrainingConfigError::MissingValue { flag })
}

fn parse_usize(value: &str, flag: &'static str) -> Result<usize, TrainingConfigError> {
    value
        .parse::<usize>()
        .map_err(|_| TrainingConfigError::InvalidInteger {
            flag,
            value: value.to_owned(),
        })
}

fn parse_u64(value: &str, flag: &'static str) -> Result<u64, TrainingConfigError> {
    value
        .parse::<u64>()
        .map_err(|_| TrainingConfigError::InvalidInteger {
            flag,
            value: value.to_owned(),
        })
}

fn parse_f32(value: &str, flag: &'static str) -> Result<f32, TrainingConfigError> {
    value
        .parse::<f32>()
        .map_err(|_| TrainingConfigError::InvalidFloat {
            flag,
            value: value.to_owned(),
        })
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
            config.train,
            DatasetPaths {
                images: PathBuf::from("train-images-idx3-ubyte"),
                labels: PathBuf::from("train-labels-idx1-ubyte"),
            }
        );
        assert_eq!(config.epochs, DEFAULT_EPOCHS);
        assert_eq!(config.learning_rate, DEFAULT_LEARNING_RATE);
        assert_eq!(config.graph_node_count, DEFAULT_GRAPH_NODE_COUNT);
        assert_eq!(config.graph_search_limit, None);
        assert_eq!(config.run_dir, None);
        assert_eq!(config.checkpoint_every, None);
        assert_eq!(config.eval_every, DEFAULT_EVAL_EVERY);
        assert_eq!(config.test, None);
        assert!(config.distributed_workers.is_empty());
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
            TrainingConfigError::MissingRequiredArgument {
                flag: "--train-labels"
            }
        ));
    }

    #[test]
    fn parses_experiment_management_flags() {
        let config = TrainingConfig::from_args([
            "golem-engine".to_owned(),
            "--train-images".to_owned(),
            "train-images-idx3-ubyte".to_owned(),
            "--train-labels".to_owned(),
            "train-labels-idx1-ubyte".to_owned(),
            "--test-images".to_owned(),
            "t10k-images-idx3-ubyte".to_owned(),
            "--test-labels".to_owned(),
            "t10k-labels-idx1-ubyte".to_owned(),
            "--run-dir".to_owned(),
            "runs/manual".to_owned(),
            "--checkpoint-every".to_owned(),
            "5".to_owned(),
            "--eval-every".to_owned(),
            "2".to_owned(),
            "--load-checkpoint".to_owned(),
            "input.json".to_owned(),
            "--save-checkpoint".to_owned(),
            "output.json".to_owned(),
        ])
        .expect("parse experiment config");

        assert_eq!(config.run_dir, Some(PathBuf::from("runs/manual")));
        assert_eq!(config.checkpoint_every, Some(5));
        assert_eq!(config.eval_every, 2);
        assert_eq!(config.load_checkpoint, Some(PathBuf::from("input.json")));
        assert_eq!(config.save_checkpoint, Some(PathBuf::from("output.json")));
        assert_eq!(
            config.test,
            Some(DatasetPaths {
                images: PathBuf::from("t10k-images-idx3-ubyte"),
                labels: PathBuf::from("t10k-labels-idx1-ubyte"),
            })
        );
    }

    #[test]
    fn parses_distributed_worker_addresses() {
        let config = TrainingConfig::from_args([
            "golem-engine".to_owned(),
            "--train-images".to_owned(),
            "train-images-idx3-ubyte".to_owned(),
            "--train-labels".to_owned(),
            "train-labels-idx1-ubyte".to_owned(),
            "--distributed-worker".to_owned(),
            "worker-a:7000".to_owned(),
            "--distributed-worker".to_owned(),
            "worker-b:7000".to_owned(),
        ])
        .expect("parse distributed config");

        assert_eq!(
            config.distributed_workers,
            vec!["worker-a:7000".to_owned(), "worker-b:7000".to_owned()]
        );
    }

    #[test]
    fn parses_worker_server_mode() {
        let command = CliCommand::from_args([
            "golem-engine".to_owned(),
            "--worker-listen".to_owned(),
            "0.0.0.0:7000".to_owned(),
        ])
        .expect("parse worker command");

        match command {
            CliCommand::Worker(config) => assert_eq!(config.listen_addr, "0.0.0.0:7000"),
            CliCommand::Train(_) => panic!("expected worker mode"),
        }
    }
}
