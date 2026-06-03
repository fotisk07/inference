import torch
import lightning as L
from transformers import VisionEncoderDecoderModel, DonutProcessor, get_linear_schedule_with_warmup


class DonutModule(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        processor: DonutProcessor,
        additional_tokens: list[str],
        learning_rate: float = 1e-4,
        warmup_steps: int = 300,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["processor"])
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        # Register any new tokens (task start token + domain tokens like <champ>)
        if additional_tokens:
            processor.tokenizer.add_special_tokens({"additional_special_tokens": additional_tokens})
            self.model.decoder.resize_token_embeddings(len(processor.tokenizer))

        self.model.config.pad_token_id = processor.tokenizer.pad_token_id
        self.model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
            additional_tokens[0] if additional_tokens else processor.tokenizer.bos_token
        )

    def forward(self, pixel_values, labels):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch["pixel_values"], batch["labels"])
        self.log("train_loss", outputs.loss, prog_bar=True, on_step=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["pixel_values"], batch["labels"])
        self.log("val_loss", outputs.loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
