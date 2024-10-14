import os
import subprocess

from transformers import AutoModelForCausalLM

from src.constants.llm import LLM_MODEL
from src.utils.data import get_file_system


# Function to cache a model from Hugging Face (HF) hub and store it locally or on S3 (if needed)
def cache_model_from_hf_hub(
    model_name,
    s3_bucket="projet-dedup-oja",  # Default S3 bucket name
    s3_cache_dir="models/hf_hub",   # Default directory on S3 for storing cached models
):
    """
    Use S3 as a proxy cache for Hugging Face models if the model is not already cached locally.
    
    Args:
        model_name (str): Name of the model on Hugging Face hub.
        s3_bucket (str): Name of the S3 bucket to use for storage.
        s3_cache_dir (str): Path of the cache directory on the S3 bucket.
    """
    
    # Local cache configuration
    LOCAL_HF_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    # Convert model name to a format that matches the local Hugging Face cache format
    model_name_hf_cache = "models--" + "--".join(model_name.split("/"))
    dir_model_local = os.path.join(LOCAL_HF_CACHE_DIR, model_name_hf_cache)

    # Remote (S3) cache configuration
    fs = get_file_system()  # Retrieve the file system utility to interact with S3
    # List all models available in the S3 cache directory
    available_models_s3 = [
        os.path.basename(path) for path in fs.ls(os.path.join(s3_bucket, s3_cache_dir))
    ]
    dir_model_s3 = os.path.join(s3_bucket, s3_cache_dir, model_name_hf_cache)

    # If the model is not already cached locally
    if model_name_hf_cache not in os.listdir(LOCAL_HF_CACHE_DIR):
        # Check if the model exists in the S3 cache
        if model_name_hf_cache in available_models_s3:
            print(f"Fetching model {model_name} from S3.")
            # Use the MinIO CLI (mc) command to copy the model from S3 to the local Hugging Face cache
            cmd = [
                "mc",
                "cp",
                "-r",  # Recursive copy
                f"s3/{dir_model_s3}",  # Source (S3 directory)
                f"{LOCAL_HF_CACHE_DIR}/",  # Destination (local cache)
            ]
            subprocess.run(cmd, check=True)  # Run the command to copy the model
        else:
            # If the model isn't in S3, fetch it from the Hugging Face hub
            print(f"Model {model_name} not found on S3, fetching from HF hub.")
            # Load the model using Hugging Face’s `from_pretrained` method, which downloads it
            AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",  # Automatically selects the optimal dtype (e.g., float16)
            )
            # Once the model is downloaded, upload it to S3 for future access
            print(f"Putting model {model_name} on S3.")
            cmd = [
                "mc",
                "cp",
                "-r",  # Recursive copy
                f"{dir_model_local}/",  # Source (local cache)
                f"s3/{dir_model_s3}",  # Destination (S3 directory)
            ]
            subprocess.run(cmd, check=True)  # Run the command to upload the model to S3
    else:
        # If the model is already cached locally
        print(f"Model {model_name} found in local cache.")
        # Check if the model is missing from the S3 cache
        if model_name_hf_cache not in available_models_s3:
            # Upload the locally cached model to S3
            print(f"Putting model {model_name} on S3.")
            cmd = [
                "mc",
                "cp",
                "-r",  # Recursive copy
                f"{dir_model_local}/",  # Source (local cache)
                f"s3/{dir_model_s3}",  # Destination (S3 directory)
            ]
            subprocess.run(cmd, check=True)  # Run the command to upload the model to S3


# Main function to run the cache operation on the specified model
if __name__ == "__main__":
    cache_model_from_hf_hub(LLM_MODEL)  # Caches the model defined by LLM_MODEL
