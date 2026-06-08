FIELD_NAMES = [
    "BR/COMISSION_PAYMENT/destinataire",
    "BR/COMISSION_PAYMENT/E-mail",
    "BR/COMISSION_PAYMENT/cpf_cnpj_prestador",
    "BR/COMISSION_PAYMENT/cpf_cnpj_tomador",
    "BR/COMISSION_PAYMENT/data_emissao",
    "BR/COMISSION_PAYMENT/numero_da_nota",
    "BR/COMISSION_PAYMENT/servico_prestado",
    "BR/COMISSION_PAYMENT/valor_da_nota",
]


def _token(field_name: str) -> str:
    return f"<{field_name.split('/')[-1]}>"


class LabelFormatter:
    FIELDS = FIELD_NAMES
    TOKENS = [_token(f) for f in FIELD_NAMES]

    @classmethod
    def get_all_tokens(cls) -> list[str]:
        return cls.TOKENS

    def format(self, sample: dict) -> str:
        """
        sample["fields"] = [{"field_name": ..., "annotator_text": ..., "ocr_text": ...}, ...]
        Returns a string of present fields only:
          <field_a> value_a <field_a> <field_b> value_b <field_b> ...
        """
        values = {
            f["field_name"]: f["annotator_text"].strip() for f in sample["fields"]
        }

        parts = []
        for field_name in self.FIELDS:
            value = values.get(field_name, "")
            if value:
                tok = _token(field_name)
                parts.append(f"{tok} {value} {tok}")

        return " ".join(parts)
