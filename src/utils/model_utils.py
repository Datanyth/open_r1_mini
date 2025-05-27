from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
import torch

from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from .config import SFTConfig

def get_tokenizer(model_args: ModelConfig, training_args: SFTConfig) -> PreTrainedTokenizer:
    """
    Get the tokenizer for the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        revision=model_args.model_revision,
        # use_auth_token=model_args.use_auth_token,
    )
    
    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer

def get_model(model_args: ModelConfig, training_args: SFTConfig) -> AutoModelForCausalLM:
    """
    Get the model
    """

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )

    return model

