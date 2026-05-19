import re
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

processor = DonutProcessor.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-cord-v2"
)
model = VisionEncoderDecoderModel.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-cord-v2"
)

model.to(device)

# load document image
dataset = load_dataset("hf-internal-testing/example-documents", split="test")
image = dataset[2]["image"]

# prepare decoder inputs
task_prompt = "<s_cord-v2>"
decoder_input_ids = (
    processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")
    .to(device)
    .input_ids
)

pixel_values = processor(image, return_tensors="pt").to(device).pixel_values

outputs = model.generate(
    pixel_values.to(device),
    decoder_input_ids=decoder_input_ids.to(device),
    max_length=model.decoder.config.max_position_embeddings,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)

sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
    processor.tokenizer.pad_token, ""
)
sequence = re.sub(
    r"<.*?>", "", sequence, count=1
).strip()  # remove first task start token
print(processor.token2json(sequence))
{
    "menu": {
        "nm": "CINNAMON SUGAR",
        "unitprice": "17,000",
        "cnt": "1 x",
        "price": "17,000",
    },
    "sub_total": {"subtotal_price": "17,000"},
    "total": {"total_price": "17,000", "cashprice": "20,000", "changeprice": "3,000"},
}
