from .config import SFTConfig
from transformers import AutoConfig
from huggingface_hub import (
    create_branch,
    create_repo,
    list_repo_commits,
    list_repo_files,
    upload_folder,
    repo_exists,
    get_safetensors_metadata,
)
import logging
import re

logger = logging.getLogger(__name__)


def push_to_hub_revision(training_args: SFTConfig, extra_ignore_patterns=[]) -> None:
    """
    Push the model to the hub.
    """
    
    repo_url = create_repo(repo_id=training_args.hub_model_id, private=False, exist_ok=True)
    initial_commit = list_repo_commits(training_args.hub_model_id)[-1]
    create_branch(
        repo_id=training_args.hub_model_id,
        branch=training_args.hub_model_revision,
        revision=initial_commit.commit_id,
        exist_ok=True,
    )

    logger.info(f"Create target repo at {repo_url}")
    logger.info(f"Pushing to the hub revision {training_args.hub_model_revision}")
    ignore_patterns = ["checkpoint-*", "*.pth"]
    ignore_patterns.extend(extra_ignore_patterns)
    future = upload_folder(
        repo_id=training_args.hub_model_id,
        folder_path=training_args.output_dir,
        revision=training_args.hub_model_revision,
        commit_message=f"Add {training_args.hub_model_revision} checkpoint",
        ignore_patterns=ignore_patterns,
        run_as_future=True
    )
    logger.info(f"Pushed to {repo_url} revision {training_args.hub_model}")

    return future

def check_hub_revision_exist(training_args: SFTConfig):
    """
    Checks if a given Hub
    """    
    if repo_exists(training_args.hub_model_id):
        if training_args.push_to_hub is True:
            revisions = [rev.name for rev in list_repo_files(training_args.hub_model_id)]

            if training_args.hub_model_revision in revisions:
                repo_files = list_repo_files(
                    repo_id=training_args.hub_model_id,
                    reivision=training_args.args.hub_model_revision,
                )

                if "README.md" in repo_files and training_args.overwrite_hub_revision is False:
                    raise ValueError(
                        f"Revision {training_args.hub_model_revision} already exists."
                        "Use --overwrite_hub_revision to overwrite it."
                    )
                
def get_param_count_from_repo_id(repo_id: str) -> int:
    try:
        metadata = get_safetensors_metadata(repo_id)
        return list(metadata.parameter_count.values())[0]
    except Exception:
        pattern = r"((\d+(\.\d+)?)(x(\d+(\.\d+)?))?)([bm])"
        matches = re.findall(pattern, repo_id.lower())

        param_counts = []
        for full_math, number1, _, _, number2, _, unit in matches:
            if number2:
                number = float(number1) * float(number2)
            else:
                number = float(number1)

            if unit == "b":
                number *= 1_000_000_000
            elif unit == "m":
                number *= 1_000_000

            param_counts.append(number)
        
        if len(param_counts) > 0:
            return int(max(param_counts))
        else:
            return -1 # no pattern found
        
def get_gpu_count_for_vllm(model_name: str, revision: str = "main", num_gpus: int = 8) -> int:
    config = AutoConfig(model_name, revision=revision, trust_remote_code=True)

    num_heads = config.num_attention_heads

    while num_heads % num_gpus != 0 or 64 % num_gpus !=0:
        logger.info(f"Reducing num_gpus from {num_gpus} to {num_gpus - 1} to make num_heads divisible by num_gpus")
        num_gpus -= 1
    return num_gpus