"""Hydra construction with explicit restore-only initialization."""
from copy import deepcopy
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.models.checkpoints import require_checkpoint_version


def _construct_model(config, *, restoring):
    config = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else deepcopy(config)
    if restoring:
        encoder = config.get("net", {}).get("encoder")
        if encoder is not None:
            encoder.update(pretrained=False, source_checkpoint=None, source_checkpoint_sha256=None)
    model = instantiate(config, _convert_="all")
    # Objective settings are part of scientific checkpoint identity, initialization paths are not.
    model.contract["loss"] = config["loss"]
    return model


def build_model(config, *, checkpoint=None):
    """Construct for a new run or Lightning-managed restore without initial weights."""
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require_checkpoint_version(state)
        del state
    return _construct_model(config, restoring=bool(checkpoint))


def restore_model(config, checkpoint, *, data_contract):
    """Restore a detached model strictly, including its model and input contracts.

    Trainer-managed resume remains responsible for optimizer and loop state.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require_checkpoint_version(state)
    if state.get("data_contract") != data_contract:
        raise ValueError("Checkpoint data contract differs from the requested inputs")
    model = _construct_model(config, restoring=True)
    model.on_load_checkpoint(state)
    model.load_state_dict(state["state_dict"], strict=True)
    return model, state
