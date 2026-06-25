MODEL_ID = "naver-clova-ix/donut-base"

# ── Canonical defaults ────────────────────────────────────────────────────────
# The single source of truth for the fine-tuning resolution and decode length.
# Sweep scripts (bench_speed) override these with their own experiment ranges.
DEFAULT_IMAGE_SIZE = (1280, 960)  # (height, width) the encoder/processor run at
DEFAULT_MAX_NEW_TOKENS = 128  # inference generation budget (predict.py)
DEFAULT_MAX_LENGTH = 128  # training decoder sequence length (dataset/train)

# ── Fine-tuning vocabulary ────────────────────────────────────────────────────
# Task token used as the decoder start (canonical Donut convention) and the set
# of extraction fields. The dataset/metrics/training code registers these onto
# the processor and scores against them; see donut.dataset.register_field_tokens.
TASK_TOKEN = "<s_donut>"

# Placeholder content for an absent field (token2json format only) — lets the
# model learn an explicit "missing" signal instead of an empty span.
MISSING_TOKEN = "<missing>"

# Extraction fields. Encoded token2json-style as <s_field>value</s_field>.
FIELD_TOKENS = [
    "<numero_da_nota>",
    "<data_emissao>",
    "<cpf_cnpj_prestador>",
    "<cep_prestador>",
    "<cpf_cnpj_tomador>",
    "<servico_prestado>",
    "<valor_da_nota>",
    "<calculo_do_imposto>",
]
