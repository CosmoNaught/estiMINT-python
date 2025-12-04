"""
Model storage and persistence functions for estiMINT package.

Equivalent to: storage.R
"""

import os
import json
import hashlib
import pickle
import zipfile
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union

import numpy as np
import xgboost as xgb


def _get_package_data_dir() -> Path:
    """Get the data directory inside the installed package."""
    return Path(__file__).parent / "data"


def _get_package_inst_dir() -> Path:
    """Get the inst directory inside the installed package."""
    return Path(__file__).parent / "inst"


def _model_repo() -> str:
    """
    Get the GitHub repository for model storage.
    
    Equivalent to R's .model_repo() function.
    
    Returns
    -------
    str
        Repository name in 'owner/repo' format
    """
    return "CosmoNaught/estiMINT"


def _model_cache_dir() -> Path:
    """
    Get the user cache directory for models.
    
    Equivalent to R's .model_cache_dir() function.
    
    Returns
    -------
    Path
        Path to cache directory
    """
    try:
        import appdirs
        cache_dir = Path(appdirs.user_cache_dir("estiMINT"))
    except ImportError:
        # Fallback to home directory
        cache_dir = Path.home() / ".cache" / "estiMINT"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _models_tag() -> str:
    """
    Get the current models tag from package data.
    
    Equivalent to R's .models_tag() function.
    
    Returns
    -------
    str
        Model tag string
        
    Raises
    ------
    FileNotFoundError
        If models-tag.txt is not found
    """
    # Try package inst/ location
    inst_path = _get_package_inst_dir() / "models-tag.txt"
    if inst_path.exists():
        return inst_path.read_text().strip()
    
    # Try importlib.resources
    try:
        import importlib.resources as pkg_resources
        try:
            with pkg_resources.files("estimint").joinpath("inst/models-tag.txt").open() as f:
                return f.read().strip()
        except (TypeError, FileNotFoundError):
            pass
        try:
            with pkg_resources.files("estimint").joinpath("models-tag.txt").open() as f:
                return f.read().strip()
        except (TypeError, FileNotFoundError):
            pass
    except ImportError:
        pass
    
    # Try local development location
    for local_path in [Path("inst/models-tag.txt"), Path("src/estimint/inst/models-tag.txt")]:
        if local_path.exists():
            return local_path.read_text().strip()
    
    raise FileNotFoundError(
        "models-tag.txt missing. Publish a model and ship the tag."
    )


def _models_checksums() -> Optional[Dict[str, Any]]:
    """
    Get model checksums from package data.
    
    Equivalent to R's .models_checksums() function.
    
    Returns
    -------
    dict or None
        Dictionary with 'path' and 'md5' keys, or None if not found
    """
    import csv
    
    # Try package inst/ location
    inst_path = _get_package_inst_dir() / "models-checksums.csv"
    if inst_path.exists():
        with open(inst_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            return rows if rows else None
    
    # Try importlib.resources
    try:
        import importlib.resources as pkg_resources
        try:
            with pkg_resources.files("estimint").joinpath("inst/models-checksums.csv").open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                return rows if rows else None
        except (TypeError, FileNotFoundError):
            pass
    except ImportError:
        pass
    
    # Try local development locations
    for local_path in [Path("inst/models-checksums.csv"), Path("src/estimint/inst/models-checksums.csv")]:
        if local_path.exists():
            with open(local_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                return rows if rows else None
    
    return None


def _model_root(tag: Optional[str] = None) -> Path:
    """
    Get the root directory for models.
    
    Equivalent to R's .model_root() function.
    
    Parameters
    ----------
    tag : str, optional
        Model tag (default: from _models_tag())
        
    Returns
    -------
    Path
        Path to model root directory
    """
    if tag is None:
        tag = _models_tag()
    
    # Check for override environment variable
    override = os.environ.get("ESTIMINT_MODELS_DIR", "")
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"ESTIMINT_MODELS_DIR does not exist: {override}")
        return path
    
    return _model_cache_dir() / "models" / tag


def _ensure_models(tag: Optional[str] = None) -> Path:
    """
    Ensure models are downloaded and verified.
    
    Equivalent to R's .ensure_models() function.
    
    Parameters
    ----------
    tag : str, optional
        Model tag (default: from _models_tag())
        
    Returns
    -------
    Path
        Path to model root directory
        
    Raises
    ------
    ImportError
        If required packages are not installed
    RuntimeError
        If model checksum verification fails
    """
    if tag is None:
        tag = _models_tag()
    
    root = _model_root(tag)
    ok_marker = root / ".ok"
    
    if ok_marker.exists():
        return root
    
    try:
        import requests
    except ImportError:
        raise ImportError(
            "Please install 'requests' to download published models: "
            "pip install requests"
        )
    
    root.mkdir(parents=True, exist_ok=True)
    
    # Download from GitHub releases
    repo = _model_repo()
    zip_url = f"https://github.com/{repo}/releases/download/{tag}/{tag}.zip"
    
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
        tmp_zip = tmp_file.name
        
        response = requests.get(zip_url, stream=True)
        response.raise_for_status()
        
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
    
    # Extract zip
    with zipfile.ZipFile(tmp_zip, "r") as zip_ref:
        zip_ref.extractall(root)
    
    os.unlink(tmp_zip)
    
    # Verify checksums
    checksums = _models_checksums()
    if checksums:
        for entry in checksums:
            file_path = root / entry["path"]
            expected_md5 = entry["md5"]
            
            if not file_path.exists():
                raise RuntimeError(f"Model checksum verification failed: missing file {entry['path']}")
            
            # Calculate MD5
            with open(file_path, "rb") as f:
                actual_md5 = hashlib.md5(f.read()).hexdigest()
            
            if actual_md5 != expected_md5:
                raise RuntimeError(
                    f"Model checksum verification failed for {entry['path']}: "
                    f"expected {expected_md5}, got {actual_md5}"
                )
    
    # Mark as complete
    ok_marker.touch()
    
    return root


def _find_installed_model() -> Optional[str]:
    """
    Find model files installed with package.
    
    Returns
    -------
    str or None
        Path to model directory if found, None otherwise
    """
    data_dir = _get_package_data_dir()
    
    # Check for JSON format (exported from R)
    if (data_dir / "estiMINT_booster.json").exists():
        return str(data_dir)
    
    # Check for pickle format
    if (data_dir / "estiMINT_model.pkl").exists():
        return str(data_dir / "estiMINT_model.pkl")
    
    return None


def _resolve_model_file(dir_or_file: Union[str, Path]) -> str:
    """
    Resolve model file from directory or file path.
    
    Parameters
    ----------
    dir_or_file : str or Path
        Path to directory or file
        
    Returns
    -------
    str
        Path to model file or directory
        
    Raises
    ------
    FileNotFoundError
        If model file cannot be found
    """
    path = Path(dir_or_file)
    
    # If it's a file that exists, return it
    if path.is_file():
        return str(path)
    
    # Must be a directory
    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {dir_or_file}")
    
    # Check for JSON format (new format from R export)
    if (path / "estiMINT_booster.json").exists():
        return str(path)
    
    # Try candidate locations for pickle
    candidates = [
        path / "estiMINT_model.pkl",
        path / "eir_model" / "estiMINT_model.pkl",
    ]
    
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    
    # Search recursively
    for pattern in ["estiMINT_model.pkl", "estiMINT_booster.json"]:
        hits = list(path.rglob(pattern))
        if hits:
            if pattern.endswith(".json"):
                return str(hits[0].parent)
            return str(hits[0])
    
    raise FileNotFoundError(f"Could not find estiMINT model under: {dir_or_file}")


def _load_from_json_dir(model_dir: Path) -> Dict[str, Any]:
    """
    Load model from JSON files exported from R.
    
    Parameters
    ----------
    model_dir : Path
        Directory containing estiMINT_booster.json, estiMINT_calibrator.json, 
        estiMINT_metadata.json
        
    Returns
    -------
    dict
        estiMINT_model object
    """
    booster_path = model_dir / "estiMINT_booster.json"
    calibrator_path = model_dir / "estiMINT_calibrator.json"
    metadata_path = model_dir / "estiMINT_metadata.json"
    
    # Load XGBoost booster
    booster = xgb.Booster()
    booster.load_model(str(booster_path))
    
    # Load calibrator
    with open(calibrator_path, "r") as f:
        cal_data = json.load(f)
    
    calibrator = {
        "kind": cal_data["kind"],
        "qmap": {
            "xq": np.array(cal_data["qmap"]["xq"]),
            "yq": np.array(cal_data["qmap"]["yq"]),
        },
        "scale": cal_data["scale"]
    }
    
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Build model bundle
    model_bundle = {
        "class": metadata.get("class", "estiMINT_model"),
        "booster": booster,
        "calibrator": calibrator,
        "features": metadata["features"],
        "best_nrounds": metadata.get("best_nrounds"),
        "preprocess": metadata.get("preprocess", {}),
    }
    
    return model_bundle


def bundle_model(
    model_path: Union[str, Path],
    pkg_root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Bundle a trained model into the package for distribution.
    
    This copies the model file into the package's data directory so that
    `load_xgb_model()` can find it without any arguments.
    
    Parameters
    ----------
    model_path : str or Path
        Path to estiMINT_model.pkl file or directory containing it
    pkg_root : str or Path, optional
        Package root directory. If None, auto-detects from this file's location.
        
    Returns
    -------
    Path
        Path to the bundled model file
        
    Examples
    --------
    >>> from estimint import bundle_model
    >>> bundle_model("/path/to/estiMINT_model.pkl")
    >>> # Now load_xgb_model() works without arguments
    >>> model = load_xgb_model()
    """
    import shutil
    
    model_path = Path(model_path)
    
    # Resolve to actual .pkl file
    if model_path.is_dir():
        # Search for the model file
        candidates = [
            model_path / "estiMINT_model.pkl",
            model_path / "models" / "estiMINT_model.pkl",
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break
        if found is None:
            # Try recursive search
            hits = list(model_path.rglob("estiMINT_model.pkl"))
            if hits:
                found = hits[0]
        if found is None:
            raise FileNotFoundError(f"Could not find estiMINT_model.pkl in {model_path}")
        model_path = found
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not model_path.suffix == ".pkl":
        raise ValueError(f"Expected .pkl file, got: {model_path}")
    
    # Determine package root
    if pkg_root is None:
        # Auto-detect: this file is in src/estimint/storage.py
        # Package root is 3 levels up
        pkg_root = Path(__file__).parent.parent.parent
    else:
        pkg_root = Path(pkg_root)
    
    # Determine data directory
    # Check if we're in src layout or flat layout
    if (pkg_root / "src" / "estimint").is_dir():
        data_dir = pkg_root / "src" / "estimint" / "data"
    elif (pkg_root / "estimint").is_dir():
        data_dir = pkg_root / "estimint" / "data"
    else:
        # Assume we're inside the package itself
        data_dir = Path(__file__).parent / "data"
    
    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model file
    dest = data_dir / "estiMINT_model.pkl"
    shutil.copy2(model_path, dest)
    
    # Compute checksum
    with open(dest, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    
    # Write checksum file
    checksum_file = data_dir / "model_checksum.txt"
    checksum_file.write_text(f"{md5}  estiMINT_model.pkl\n")
    
    print(f"✓ Model bundled successfully!")
    print(f"  Source: {model_path}")
    print(f"  Destination: {dest}")
    print(f"  MD5: {md5}")
    print(f"\nNow reinstall the package:")
    print(f"  cd {pkg_root} && pip install -e .")
    print(f"\nThen load_xgb_model() will work without arguments.")
    
    return dest


def save_xgb_model(
    model_dir: Union[str, Path],
    tag: Optional[str] = None,
    pkg_root: Union[str, Path] = ".",
    repo: Optional[str] = None,
    overwrite: bool = True,
    wait_seconds: int = 90
) -> str:
    """
    Save XGBoost model to GitHub releases.
    
    Equivalent to R's save_xgb_model() function.
    
    Parameters
    ----------
    model_dir : str or Path
        Directory containing the model
    tag : str, optional
        Release tag (auto-generated if None)
    pkg_root : str or Path, optional
        Package root directory (default: ".")
    repo : str, optional
        GitHub repository (default: from _model_repo())
    overwrite : bool, optional
        Whether to overwrite existing files (default: True)
    wait_seconds : int, optional
        Seconds to wait for release creation (default: 90)
        
    Returns
    -------
    str
        The tag used for the release
        
    Raises
    ------
    FileNotFoundError
        If model_dir does not exist
    ImportError
        If required packages are not installed
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir does not exist: {model_dir}")
    
    resolved = Path(_resolve_model_file(model_dir))
    
    if repo is None:
        repo = _model_repo()
    
    pkg_root = Path(pkg_root).resolve()
    
    # Generate tag if not provided
    if tag is None:
        # Hash the booster file
        if resolved.is_dir():
            hash_file = resolved / "estiMINT_booster.json"
        else:
            hash_file = resolved
        
        with open(hash_file, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        short_md5 = md5[:8]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        tag = f"models-{timestamp}-{short_md5}"
    
    # Write inst/ files
    inst_dir = pkg_root / "inst"
    inst_dir.mkdir(exist_ok=True)
    
    (inst_dir / "models-tag.txt").write_text(tag + "\n")
    
    print(
        f"Model packaged under tag '{tag}'. "
        f"To publish, upload to GitHub releases. "
        f"Commit & reinstall to ship updated 'inst/models-*'."
    )
    
    return tag


def load_xgb_model(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load model onto memory for usage.
    
    Equivalent to R's load_xgb_model() function.
    
    Parameters
    ----------
    path : str or Path, optional
        Path to model directory (containing JSON files) or .pkl file.
        If None, tries (in order): ESTIMINT_MODELS_DIR env var,
        package data directory, then download using models-tag.txt.
        
    Returns
    -------
    dict
        An 'estiMINT_model' object (dictionary with model components)
        
    Raises
    ------
    FileNotFoundError
        If model file cannot be found
    """
    # 1) Explicit path
    if path is not None:
        path = Path(path)
        resolved = Path(_resolve_model_file(path))
        
        # Check if it's a directory with JSON files
        if resolved.is_dir() and (resolved / "estiMINT_booster.json").exists():
            return _load_from_json_dir(resolved)
        
        # Otherwise assume pickle
        if resolved.is_file() and resolved.suffix == ".pkl":
            with open(resolved, "rb") as f:
                obj = pickle.load(f)
            if not isinstance(obj, dict) or obj.get("class") != "estiMINT_model":
                warnings.warn("Loaded object does not appear to be an 'estiMINT_model'")
            return obj
        
        raise FileNotFoundError(f"Could not load model from: {path}")
    
    # 2) Check environment variable override
    override = os.environ.get("ESTIMINT_MODELS_DIR", "")
    if override:
        resolved = Path(_resolve_model_file(override))
        if resolved.is_dir() and (resolved / "estiMINT_booster.json").exists():
            return _load_from_json_dir(resolved)
        if resolved.is_file():
            with open(resolved, "rb") as f:
                return pickle.load(f)
    
    # 3) Check for installed model in package data/
    inst = _find_installed_model()
    if inst is not None:
        inst_path = Path(inst)
        if inst_path.is_dir() and (inst_path / "estiMINT_booster.json").exists():
            return _load_from_json_dir(inst_path)
        if inst_path.is_file():
            with open(inst_path, "rb") as f:
                return pickle.load(f)
    
    # 4) Download from GitHub releases
    tag = _models_tag()
    root = _model_root(tag)
    
    if not (root / ".ok").exists():
        _ensure_models(tag)
    
    resolved = Path(_resolve_model_file(root))
    if resolved.is_dir() and (resolved / "estiMINT_booster.json").exists():
        return _load_from_json_dir(resolved)
    
    with open(resolved, "rb") as f:
        obj = pickle.load(f)
    
    if not isinstance(obj, dict) or obj.get("class") != "estiMINT_model":
        warnings.warn("Loaded object does not appear to be an 'estiMINT_model'")
    
    return obj
