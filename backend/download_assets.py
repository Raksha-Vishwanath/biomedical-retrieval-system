from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import gdown


REPO_ROOT = Path(__file__).resolve().parents[1]


def _has_contents(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _download_zip(url_or_id: str, output_path: Path) -> None:
    if url_or_id.startswith("http://") or url_or_id.startswith("https://"):
        result = gdown.download(url=url_or_id, output=str(output_path), quiet=False, fuzzy=True)
    else:
        result = gdown.download(id=url_or_id, output=str(output_path), quiet=False, fuzzy=True)
    if not result:
        raise RuntimeError(f"Failed to download archive from Google Drive: {url_or_id}")


def _resolve_extracted_source(extract_root: Path, target_name: str) -> Path:
    direct_target = extract_root / target_name
    if direct_target.exists():
        return direct_target

    children = [child for child in extract_root.iterdir()]
    if len(children) == 1 and children[0].is_dir():
        nested_target = children[0] / target_name
        if nested_target.exists():
            return nested_target
        return children[0]

    return extract_root


def _sync_archive(url_or_id: str, target_name: str, force_refresh: bool) -> None:
    destination = REPO_ROOT / target_name
    if _has_contents(destination) and not force_refresh:
        print(f"Skipping {target_name}: destination already populated at {destination}")
        return

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        archive_path = tmp_dir / f"{target_name}.zip"
        extract_root = tmp_dir / "extracted"
        extract_root.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {target_name} archive from Google Drive...")
        _download_zip(url_or_id, archive_path)

        print(f"Extracting {target_name} archive...")
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(extract_root)

        source = _resolve_extracted_source(extract_root, target_name)
        if not source.exists():
            raise RuntimeError(f"Could not find extracted content for {target_name}")

        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        print(f"Installed {target_name} to {destination}")


def main() -> None:
    force_refresh = os.getenv("FORCE_ASSET_SYNC", "").lower() in {"1", "true", "yes"}
    specs = (
        ("GDRIVE_DATA_URL", "data"),
        ("GDRIVE_EVAL_URL", "eval"),
    )

    missing = [env_name for env_name, _ in specs if not os.getenv(env_name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing required environment variables for Render asset sync: {joined}"
        )

    for env_name, target_name in specs:
        _sync_archive(os.environ[env_name], target_name, force_refresh=force_refresh)


if __name__ == "__main__":
    main()
