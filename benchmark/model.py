from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel


@dataclass
class PreprocessResult:
    pixel_values: torch.Tensor
    decoder_input_ids: torch.Tensor
    orig_image_width_mean: float
    orig_image_height_mean: float
    orig_megapixels_mean: float
    processed_image_width: int
    processed_image_height: int
    processed_megapixels: float


@dataclass
class ModelBundle:
    processor: DonutProcessor
    model: VisionEncoderDecoderModel
    device: str
    task_prompt: str

    @classmethod
    def load(cls, model_id: str, device: str, task_prompt: str) -> ModelBundle:
        processor = DonutProcessor.from_pretrained(model_id)
        model = VisionEncoderDecoderModel.from_pretrained(model_id)
        model.to(device)
        model.eval()
        return cls(
            processor=processor, model=model, device=device, task_prompt=task_prompt
        )

    def preprocess(self, images: list[Image.Image]) -> PreprocessResult:
        """Return preprocessing result including pixel tensors and image dimension metadata.

        DonutImageProcessor resizes all images to the same fixed spatial size,
        so no pixel-level padding is needed. The same task prompt is repeated
        for each image in the batch.
        """
        widths = [img.width for img in images]
        heights = [img.height for img in images]
        orig_w_mean = sum(widths) / len(widths)
        orig_h_mean = sum(heights) / len(heights)
        orig_mp_mean = sum(w * h for w, h in zip(widths, heights)) / len(images) / 1e6

        pixel_values = self.processor(images, return_tensors="pt").pixel_values.to(
            self.device
        )
        proc_h, proc_w = pixel_values.shape[2], pixel_values.shape[3]
        proc_mp = proc_w * proc_h / 1e6

        single_ids = self.processor.tokenizer(
            self.task_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids
        decoder_input_ids = single_ids.repeat(len(images), 1).to(self.device)

        return PreprocessResult(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            orig_image_width_mean=orig_w_mean,
            orig_image_height_mean=orig_h_mean,
            orig_megapixels_mean=orig_mp_mean,
            processed_image_width=proc_w,
            processed_image_height=proc_h,
            processed_megapixels=proc_mp,
        )

    def encode(self, pixel_values: torch.Tensor):
        """Run the encoder directly and return encoder_outputs (DonutSwinModelOutput)."""
        with torch.no_grad():
            return self.model.encoder(pixel_values, return_dict=True)

    def decode(
        self,
        pixel_values: torch.Tensor,
        encoder_outputs,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run autoregressive generation with pre-computed encoder outputs.

        Passing encoder_outputs causes HuggingFace generate() to skip the
        encoder forward pass entirely. pixel_values is still passed so
        generate() can infer batch size and device.
        """
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                max_length=self.model.decoder.config.max_position_embeddings,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
        return outputs.sequences

    def count_tokens(
        self, sequences: torch.Tensor, prompt_len: int
    ) -> tuple[list[int], int, int, float]:
        """Count actual generated tokens per sample, excluding prompt and padding.

        Returns (actual_lens, max_len, sum_len, decoder_efficiency).
        decoder_efficiency = sum_len / (B * max_len).  At B=1 this is always
        1.0.  For diverse batches it reveals how much decoder compute was
        wasted on padding shorter sequences to match the longest one.
        """
        eos_id = self.processor.tokenizer.eos_token_id
        actual_lens: list[int] = []
        for b in range(sequences.shape[0]):
            row = sequences[b, prompt_len:]
            eos_positions = (row == eos_id).nonzero(as_tuple=True)[0]
            length = (
                int(eos_positions[0].item()) + 1 if len(eos_positions) > 0 else len(row)
            )
            actual_lens.append(max(length, 1))
        max_len = max(actual_lens)
        sum_len = sum(actual_lens)
        b_size = sequences.shape[0]
        efficiency = sum_len / (b_size * max_len)
        return actual_lens, max_len, sum_len, efficiency

    def postprocess(self, sequences: torch.Tensor) -> list[tuple[str, dict | None]]:
        """Decode each sequence in the batch individually.

        Returns a list of (raw_str, parsed_dict_or_None) — one entry per
        sample in the batch.
        """
        results: list[tuple[str, dict | None]] = []
        for seq in sequences:
            text = self.processor.batch_decode(seq.unsqueeze(0))[0]
            text = text.replace(self.processor.tokenizer.eos_token, "").replace(
                self.processor.tokenizer.pad_token, ""
            )
            text = re.sub(r"<.*?>", "", text, count=1).strip()
            try:
                parsed: dict | None = self.processor.token2json(text)
            except Exception:
                parsed = None
            results.append((text, parsed))
        return results
