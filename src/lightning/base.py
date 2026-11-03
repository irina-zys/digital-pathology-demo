"""Shared Lightning hooks; computation belongs to the injected network/loss."""
import lightning as L
from src.models.checkpoints import CHECKPOINT_VERSION, require_checkpoint_version


class BaseModule(L.LightningModule):
    def __init__(self, net, loss, optimizer):
        super().__init__()
        self.net, self.loss = net, loss
        self.optimizer_factory = optimizer
        self.task = net.task

    def forward(self, inputs):
        return self.net(inputs)

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(params=(p for p in self.parameters() if p.requires_grad))
        if any(group["lr"] <= 0 or group.get("weight_decay", 0) < 0 for group in optimizer.param_groups):
            raise ValueError("Invalid optimizer learning rate or weight decay")
        return optimizer

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def validate_data(self, data):
        if self.task != data.hparams.task:
            raise ValueError("Model and data tasks differ; select a matching experiment")
        encoder = getattr(self.net, "encoder", None)
        size = getattr(encoder, "required_input_size", None)
        if size is not None and data.contract.get("image_size") != size:
            raise ValueError(f"Configured encoder requires data.image_size={size}")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["pathology_checkpoint_version"] = CHECKPOINT_VERSION
        checkpoint["model_contract"] = self.contract
        checkpoint["data_contract"] = self.trainer.datamodule.contract

    def on_load_checkpoint(self, checkpoint):
        require_checkpoint_version(checkpoint)
        if checkpoint.get("model_contract") != self.contract:
            raise ValueError("Checkpoint model/label contract differs from the configured model")
        if self._trainer is not None and self.trainer.datamodule is not None:
            if checkpoint.get("data_contract") != self.trainer.datamodule.contract:
                raise ValueError("Checkpoint data contract differs (task/classes/image size/normalization)")
