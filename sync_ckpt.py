import os
from typing import Optional, List
from huggingface_hub import HfApi, upload_folder


def upload_folder_to_hf(
    local_folder: str,
    repo_id: str,
    repo_type: str = "model",
    commit_message: str = "Upload folder",
    path_in_repo: str = ".",
    token: Optional[str] = None,
    create_repo: bool = False,
    ignore_patterns: Optional[List[str]] = None,
) -> None:
    """
    Upload a local folder to an existing (or optionally new) Hugging Face Hub repo.

    Args:
        local_folder: Path to the local folder to upload.
        repo_id: Target repo id, e.g. "username/my-repo".
        repo_type: One of {"model", "dataset", "space"}. Default: "model".
        commit_message: Commit message for the upload.
        path_in_repo: Subdirectory in the repo where files will be placed. Default: "." (repo root).
        token: HF token. If None, uses cached login from HfFolder or HF_TOKEN env.
        create_repo: If True, create the repo if it doesn't exist.
        ignore_patterns: List of glob patterns to exclude from upload.
    """

    if not os.path.isdir(local_folder):
        raise FileNotFoundError(f"Local folder does not exist: {local_folder}")

    # Resolve token: explicit > env(HF_TOKEN) > cached login
    token = token or os.environ.get("HF_TOKEN") 
    if token is None:
        raise RuntimeError(
            "No HF token found. Run 'huggingface-cli login' or pass --token / set HF_TOKEN."
        )

    api = HfApi()

    if create_repo:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, token=token)

    # Perform upload in a single call (no local git clone required)
    upload_folder(
        repo_id=repo_id,
        folder_path=local_folder,
        path_in_repo=path_in_repo,
        repo_type=repo_type,
        commit_message=commit_message,
        token=token,
        ignore_patterns=ignore_patterns,
    )

    print(f"Uploaded '{local_folder}' to hf://{repo_type}s/{repo_id}/{path_in_repo}")



if __name__ == "__main__":
    uploaded_models = {
        # "checkpoints_torch/solaris.pt": "checkpoints/solaris.pt",
        # "checkpoints_torch/clip.pt": "checkpoints/clip.pt",
        # "checkpoints_torch/vae.pt": "checkpoints/vae.pt",
        "checkpoints_torch/": "checkpoints/"
    }
    repo_id = "quangngcs/lmw"
    repo_type = "model"
    commit_message = "Upload folder"
    token = None
    create_repo = False
    # Example: ignore caches and temp/log files
    ignore_patterns = ["**/__pycache__/**", "**/*.tmp", "**/*.log", "**/.DS_Store", "*.bin", "*.pkl"]
    
    # submit_folder = "training_runs/"

    for model, path_in_repo in uploaded_models.items():
        # local_folder = os.path.join(submit_folder, model)
        local_folder = model
        upload_folder_to_hf(
            local_folder=local_folder,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
            path_in_repo=path_in_repo,
        )

    # for local_folder, path_in_repo in uploaded_models.items():
    #     upload_folder_to_hf(
    #         local_folder=local_folder,
    #         repo_id=repo_id,
    #         repo_type=repo_type,
    #         commit_message=commit_message,
    #         path_in_repo=path_in_repo,
    #         token=token,
    #         create_repo=create_repo,
    #         ignore_patterns=ignore_patterns,
    #     )
