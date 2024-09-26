import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def build_llm_model(
    model_name,
    hf_token=None,
):
    """
    Create the llm model
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=hf_token)

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    # Load LLM tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        use_fast=True,
    )

    # Check if tokenizer has a pad_token; if not, set it to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        config=config,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )

    return model, tokenizer
