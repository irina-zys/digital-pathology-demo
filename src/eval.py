"""Evaluate a trusted local checkpoint: python -m src.eval ckpt_path=..."""

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig

from src.train import run


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg):
    run(cfg, Path(HydraConfig.get().runtime.output_dir), evaluate=True)


if __name__ == "__main__":
    main()
