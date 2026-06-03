from torch.utils.data import Dataset
from transformers import DonutProcessor


class DonutDataset(Dataset):
    def __init__(
        self,
        data,
        processor: DonutProcessor,
        label_formatter,
        task_start_token: str,
        max_target_length: int = 512,
    ):
        self.data = data
        self.processor = processor
        self.label_formatter = label_formatter
        self.task_start_token = task_start_token
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        image = sample["image"].convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        target_text = self.task_start_token + self.label_formatter.format(sample)
        tokenized = self.processor.tokenizer(
            target_text,
            add_special_tokens=False,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = tokenized.input_ids.squeeze(0).clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}
