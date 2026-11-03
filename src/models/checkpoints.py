"""Version boundary for the modular training checkpoint layout."""
CHECKPOINT_VERSION = 2


def require_checkpoint_version(checkpoint):
    if checkpoint.get("pathology_checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Incompatible historical checkpoint: modular models require checkpoint version 2; retrain with the current configuration")
