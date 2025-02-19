
from rl_zoo3.hyperparams_opt import optimize_hyperparameters

optimize_hyperparameters(
    "ISOEnv-v0",
    "PPO",
    n_trials=10,
    n_timesteps=10,
    n_jobs=4,
    sampler_method="tpe",
    pruner_method="median",
    n_startup_trials=10,
    n_evaluations=2,
    seed=42,
)