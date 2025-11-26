from huggingface_hub import HfApi

api = HfApi()

api.upload_large_folder(
    folder_path="2DJigsaw",          # your local dataset folder
    repo_id="JXZhou0224/2DJigsaw",
    repo_type="dataset"
)
