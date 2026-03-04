from huggingface_hub import snapshot_download
import os


def download_hf_repo(repo_id, local_dir, repo_type="model", token=None, revision=None):
    """
    Download a repository from Hugging Face Hub to a local directory.
    
    Args:
        repo_id (str): Repository ID (e.g., "microsoft/DialoGPT-medium")
        local_dir (str): Local directory path to download to
        repo_type (str): Type of repository ("model", "dataset", or "space")
        token (str, optional): Hugging Face token for private repos
        revision (str, optional): Git revision (branch, tag, or commit hash)
    """
    print(f"Downloading {repo_type} {repo_id} to {local_dir}...")
    
    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type=repo_type,
            local_dir_use_symlinks=False,  # Download actual files instead of symlinks
            # ignore_patterns=["raw_labels/*"],
            # allow_patterns=["FiFA-100k/*"],
            max_workers=32
        )
        print(f"Successfully downloaded {repo_id} to {local_dir}")
       
    except Exception as e:
        print(f"Error downloading {repo_id}: {str(e)}")
        raise


def main(args):
    # Configuration - modify these variables as needed

    # repo_id = args.repo_id
    # local_dir = args.local_dir
    repo_type = "nyu-visionx/solaris"
    token = None  # Set to your HF token if needed for private repos
    revision = None  # Set to specific branch/tag/commit if needed
    
    # Download the repository
    download_hf_repo(
        repo_id=repo_type,
        local_dir="checkpoints/",
        repo_type="model",
    )


if __name__ == "__main__":
    from omegaconf import OmegaConf
    args = OmegaConf.from_cli()
    main(args)
