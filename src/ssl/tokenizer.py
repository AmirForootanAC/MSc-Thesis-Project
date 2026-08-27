from transformers import AutoTokenizer


class ClinicalTokenizer:

    def __init__(
        self,
        model_name="distilbert-base-uncased",
        max_length=256,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        self.max_length = max_length


    def __call__(self, texts):

        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )