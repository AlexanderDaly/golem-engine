use std::env;
use std::process;

use golem_engine::training::{print_usage, run_training, TrainingConfig, TrainingConfigError};

fn main() {
    match TrainingConfig::from_args(env::args()) {
        Ok(config) => {
            if let Err(error) = run_training(config) {
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
