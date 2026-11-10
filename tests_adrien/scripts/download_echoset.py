from huggingface_hub import snapshot_download

target_dir = "/idiap/temp/adubois/EchoSet"

snapshot_download(
    repo_id="JusperLee/EchoSet",
    repo_type="dataset",
    local_dir=target_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=4  # accélère le téléchargement des fichiers en parallèle
)
