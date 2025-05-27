import os
from trl import SFTTrainer

def init_wandb_training(training_args: SFTTrainer):
    """
    Helper function for setting up Weight and Biases logging tools.
    """

    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    if training_args.wandb_run_group is not None:
        os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group