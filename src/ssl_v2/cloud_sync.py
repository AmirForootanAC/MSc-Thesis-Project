"""
Live Cloud Persistence for Kaggle 12-Hour Sessions (GitHub + Hugging Face / Kaggle).

1. Pushes lightweight JSON results/histories directly to GitHub after each stage/epoch.
2. Syncs heavy .pt checkpoints (best_ssl_model.pt, latest_ssl_checkpoint.pt, etc.)
   to/from a Private Hugging Face Model Repo so zero training progress is lost
   if a Kaggle 12-hour session times out.
"""

import os
import shutil
import subprocess
from pathlib import Path


class KagglePersistenceManager:
    """
    Handles automatic checkpoint restoration at session start and
    live backup during training.
    """

    def __init__(
        self,
        project_root: Path,
        hf_repo_id: str | None = None,
        hf_token: str | None = None,
        kaggle_input_backup_dir: str | None = None,
    ):
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / "results" / "ssl_v2"

        # Read from environment variables if not explicitly passed
        self.hf_repo_id = hf_repo_id or os.environ.get("HF_REPO_ID", "").strip() or None
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "").strip() or None
        self.kaggle_backup_dir = (
            Path(kaggle_input_backup_dir)
            if kaggle_input_backup_dir
            else (
                Path(os.environ["KAGGLE_BACKUP_INPUT_DIR"])
                if os.environ.get("KAGGLE_BACKUP_INPUT_DIR")
                else None
            )
        )

        self.hf_api = None
        if self.hf_repo_id and self.hf_token:
            try:
                from huggingface_hub import HfApi
                self.hf_api = HfApi(token=self.hf_token)
                self.hf_api.create_repo(
                    repo_id=self.hf_repo_id,
                    repo_type="model",
                    private=True,
                    exist_ok=True,
                )
                print(f"[CloudSync] Connected to private HF repo: {self.hf_repo_id}")
            except Exception as exc:
                print(f"[CloudSync Warning] Could not initialize HF API: {exc}")
                self.hf_api = None

    def restore_all_from_backup(self) -> None:
        """
        Call once at the start of a Kaggle session to restore any existing
        checkpoints from either a linked Kaggle Input Dataset or Hugging Face Hub.
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 1. Restore from linked previous Kaggle Notebook Output / Dataset if mounted
        if self.kaggle_backup_dir and self.kaggle_backup_dir.exists():
            print(f"[CloudSync] Restoring checkpoints from Kaggle Input: {self.kaggle_backup_dir}")
            for src_file in self.kaggle_backup_dir.rglob("*"):
                if src_file.is_file():
                    rel = src_file.relative_to(self.kaggle_backup_dir)
                    dst_file = self.results_dir / rel
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    if not dst_file.exists():
                        shutil.copy2(src_file, dst_file)
                        print(f"  -> Restored: {rel}")

        # 2. Restore from Hugging Face Private Repo if configured
        if self.hf_api is not None:
            try:
                from huggingface_hub import snapshot_download
                print(f"[CloudSync] Downloading latest checkpoints from {self.hf_repo_id}...")
                snapshot_download(
                    repo_id=self.hf_repo_id,
                    repo_type="model",
                    local_dir=str(self.results_dir),
                    token=self.hf_token,
                )
                print("[CloudSync] Restore from Hugging Face complete.")
            except Exception as exc:
                print(f"[CloudSync Info] No remote snapshot restored yet: {exc}")

    def backup_file(self, file_path: Path) -> None:
        """
        Uploads a single checkpoint or JSON file to the private HF repo immediately.
        """
        file_path = Path(file_path)
        if not file_path.exists() or self.hf_api is None:
            return

        try:
            rel_path = file_path.relative_to(self.results_dir).as_posix()
        except ValueError:
            rel_path = file_path.name

        try:
            self.hf_api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=rel_path,
                repo_id=self.hf_repo_id,
                repo_type="model",
            )
            print(f"  [CloudSync] Backed up '{rel_path}' to cloud.")
        except Exception as exc:
            print(f"  [CloudSync Warning] Upload failed for {rel_path}: {exc}")

    def push_jsons_to_github(self, commit_message: str = "Update SSL v2 JSON metrics") -> None:
        """
        Commits and pushes lightweight .json result files directly to GitHub.
        Never stages .pt files.
        """
        try:
            json_files = list(self.results_dir.rglob("*.json"))
            if not json_files:
                return

            for jf in json_files:
                subprocess.run(
                    ["git", "add", "-f", str(jf)],
                    cwd=self.project_root,
                    check=False,
                )

            subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=self.project_root,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            res = subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
            if res.returncode == 0:
                print(f"  [GitHub Sync] Pushed latest JSON metrics to GitHub ({commit_message}).")
        except Exception as exc:
            print(f"  [GitHub Sync Warning] {exc}")