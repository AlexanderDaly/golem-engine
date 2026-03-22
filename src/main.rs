use std::env;
use std::process;

use golem_engine::training::{
    print_usage, run_training, run_worker_server, CliCommand, TrainingConfigError,
};

fn main() {
    match CliCommand::from_args(env::args()) {
        Ok(CliCommand::Train(config)) => {
            if let Err(error) = run_training(config) {
                eprintln!("error: {error}");
                process::exit(1);
            }
        }
        Ok(CliCommand::Worker(config)) => {
            if let Err(error) = run_worker_server(config) {
                eprintln!("error: {error}");
                process::exit(1);
            }
        }
        Err(TrainingConfigError::HelpRequested { program }) => {
            print_usage(&program);
        }
        Err(error) => {
            eprintln!("error: {error}");
            process::exit(1);
        }
    }
}
