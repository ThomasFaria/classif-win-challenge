from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


def build_llm_model(
    model_name,
    quantization_config: bool = False,
    config: bool = False,
    token=None,
    generation_args: dict = None,
    device="auto",
):
    """
    Create the llm model
    """

    if generation_args is None:
        generation_args = {}

    configs = {
        # Load quantization config
        "quantization_config": (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=False,
            )
            if quantization_config
            else None
        ),
        # Load LLM config
        "config": (
            AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=token)
            if config
            else None
        ),
        "token": token,
    }

    # Load LLM tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, device_map=device, token=configs["token"]
    )

    # Check if tokenizer has a pad_token; if not, set it to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LLM
    model = AutoModelForCausalLM.from_pretrained(model_name, **configs)

    # Create a pipeline with  tokenizer and model
    pipeline_HF = pipeline(
        task="text-generation",  # TextGenerationPipeline HF pipeline
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=generation_args.get("max_new_tokens", 2000),
        return_full_text=generation_args.get("return_full_text", False),
        device_map=device,
        do_sample=generation_args.get("do_sample", True),
        temperature=generation_args.get("temperature", 0.2),
    )
    llm = HuggingFacePipeline(pipeline=pipeline_HF)
    return llm, tokenizer
