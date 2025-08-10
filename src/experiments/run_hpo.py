import os
import argparse
import mlflow
import optuna
from pathlib import Path


from finetuner import FineTuner
from utils import load_config, get_logger

logger = get_logger(__name__)


def objective(trial: optuna.trial.Trial, base_config: dict) -> float:
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object to sample hyperparameters.
        base_config (Dict[str, Any]): Base configuration dictionary for training.

    Returns:
        float: The evaluation loss for the trial (to be minimized).
    """
    # Sample HPO params
    config = base_config.copy()
    config["training"]["learning_rate"] = trial.suggest_float(
        "learning_rate", 5e-6, 5e-4, log=True
    )

    # Start a fresh MLflow run for this trial
    with mlflow.start_run(run_name=f"trial_{trial.number}") as run:
        mlflow.log_params(trial.params)

        # Train
        tuner = FineTuner(config)
        tuner.tune()

        try:
            metrics = tuner.evaluate()
            eval_loss = metrics["eval_loss"]
        except ValueError as e:
            # No eval_loss available: fallback to last train loss
            logger.warning("No eval set: falling back to final train loss")
            history = tuner.trainer.state.log_history
            # find last logged 'loss'
            train_losses = [x["loss"] for x in history if "loss" in x]
            eval_loss = train_losses[-1]
        mlflow.log_metric("hpo_eval_loss", eval_loss)

        # Save this trial’s checkpoint under a unique folder
        trial_ckpt = Path(config["training"]["output_dir"]) / f"trial_{trial.number}"
        trial_ckpt.mkdir(parents=True, exist_ok=True)
        tuner.save_model(trial_ckpt)

        # Log the checkpoint path as an MLflow artifact
        mlflow.log_artifacts(trial_ckpt, artifact_path="checkpoint")

        # Record it back in the Optuna trial
        trial.set_user_attr("checkpoint", trial_ckpt)
        trial.set_user_attr("mlflow_run_id", run.info.run_id)

    return eval_loss


def main(config_path: str) -> None:
    """
    Main entry point to run hyperparameter optimization (HPO) with Optuna and MLflow.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    config = load_config(config_path)
    hpo_conf = config["hpo"]

    mlflow.set_tracking_uri(os.path.abspath("./mlruns"))
    mlflow.set_experiment(config["experiment_name"])
    mlflow.pytorch.autolog(log_models=False)

    # Run Optuna
    study = optuna.create_study(direction=hpo_conf["direction"])
    study.optimize(lambda t: objective(t, config), n_trials=hpo_conf["n_trials"])

    # After HPO, get best trial
    best = study.best_trial
    best_ckpt = best.user_attrs.get("checkpoint")
    best_run_id = best.user_attrs.get("mlflow_run_id")

    logger.info(f"Best trial #{best.number} with loss {best.value}")
    logger.info(f" → Local checkpoint: {best_ckpt}")
    logger.info(f" → MLflow run ID: {best_run_id}")

    # Summarize best in MLflow
    with mlflow.start_run(run_name="best_trial_summary", nested=False):
        mlflow.log_params(best.params)
        mlflow.log_metric("best_eval_loss", best.value)
        mlflow.log_param("best_checkpoint", best_ckpt)

    # Generate and log Optuna plots
    try:
        fig1 = optuna.visualization.plot_optimization_history(study)
        mlflow.log_figure(fig1, "opt_history.png")
        fig2 = optuna.visualization.plot_param_importances(study)
        mlflow.log_figure(fig2, "param_importances.png")
    except Exception as e:
        logger.warning(f"Could not plot HPO: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Hyperparameter Optimization with Optuna and SFTTrainer."
    )
    parser.add_argument(
        "--config-path", required=True, help="Path to the HPO config YAML file."
    )
    args = parser.parse_args()
    main(args.config_path)
