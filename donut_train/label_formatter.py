FIELD_NAMES = [
    "BR/COMISSION_PAYEMENT/calculo_do_imposto",
    "BR/COMISSION_PAYEMENT/cep_prestador",
    "BR/COMISSION_PAYEMENT/cpf_cnpj_prestador",
    "BR/COMISSION_PAYEMENT/cpf_cnpj_tomador",
    "BR/COMISSION_PAYEMENT/data_emissao",
    "BR/COMISSION_PAYEMENT/numero_da_nota",
    "BR/COMISSION_PAYEMENT/servico_prestado",
    "BR/COMISSION_PAYEMENT/valor_da_nota",
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
        values = {f["field_name"]: f["annotator_text"].strip() for f in sample["fields"]}

        parts = []
        for field_name in self.FIELDS:
            value = values.get(field_name, "")
            if value:
                tok = _token(field_name)
                parts.append(f"{tok} {value} {tok}")

        return " ".join(parts)
