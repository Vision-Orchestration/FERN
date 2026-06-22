"""Load training config from YAML and merge with argparse defaults."""

import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    """Load a YAML config file and flatten into dotted keys.
    
    Example YAML:
        training:
            epochs: 100
            lr: 3e-4
    
    Returns: {"epochs": 100, "lr": 3e-4, ...}
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(p) as f:
        raw = yaml.safe_load(f)

    flat = {}
    for section, values in raw.items():
        if isinstance(values, dict):
            for k, v in values.items():
                flat[k] = v
        else:
            flat[section] = values
    return flat


def merge_config(args, config_path: str = None) -> None:
    """Overwrite parsed args with values from a YAML config file.
    
    Usage:
        args = parse_args()
        merge_config(args)          # auto-detect configs/train_config.yaml
    """
    if config_path is None:
        root = Path(__file__).resolve().parent.parent
        candidates = [
            root / "configs" / "train_config.yaml",
            root / "train_config.yaml",
        ]
        for c in candidates:
            if c.exists():
                config_path = str(c)
                break

    if config_path is None or not Path(config_path).exists():
        return

    cfg = load_config(config_path)
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)


def load_train_config(path: str = None) -> dict:
    """Shortcut: load train config, return flat dict with defaults."""
    cfg = {}
    if path is None:
        root = Path(__file__).resolve().parent.parent
        p = root / "configs" / "train_config.yaml"
        if not p.exists():
            return cfg
        path = str(p)
    if Path(path).exists():
        cfg = load_config(path)
    return cfg